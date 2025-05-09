import logging
import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection
from src.metrics import MR, minADE, minFDE, mulADE, MVNLoss
from src.metrics.prediction_avg_ade import PredAvgADE
from src.metrics.prediction_avg_fde import PredAvgFDE
from src.optim.warmup_cos_lr import WarmupCosLR

from .loss.esdf_collision_loss import ESDFCollisionLoss
from pytorch_lightning.utilities import grad_norm
from src.utils.min_norm_solvers import MinNormSolver

logger = logging.getLogger(__name__)

# torch.set_float32_matmul_precision('high')
class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        use_collision_loss=True,
        use_contrast_loss=False,
        regulate_yaw=False,
        objective_aggregate_mode: str = "mean",
        mul_ade_loss: list[str] = ['phase_loss', 'scale_loss'],
        dynamic_weight: bool = True,
        max_horizon: int = 10,
        use_dwt: bool = False,
        learning_output: str = 'velocity',
        init_weights: list[float] = [1, 1, 1, 1, 1, 1],
        wavelet: list[str] = ['cgau1', 'constant', 'bior1.3', 'constant'],
        wtd_with_history: bool = False,
        approximation_norm: bool = False,
        time_decay: bool = False,
        time_norm: bool = False,
    ) -> None:
        """
        Initializes the class.

        :param model: pytorch model
        :param objectives: list of learning objectives used for supervision at each step
        :param metrics: list of planning metrics computed at each step
        :param batch_size: batch_size taken from dataloader config
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models.
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param warm_up_lr_scheduler: config for instantiating warm up lr scheduler. Can be 'None' for older models and when a warm up lr_scheduler is not being used.
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.objective_aggregate_mode = objective_aggregate_mode
        self.history_steps = model.history_steps
        self.use_collision_loss = use_collision_loss
        self.use_contrast_loss = use_contrast_loss
        self.regulate_yaw = regulate_yaw

        self.radius = model.radius
        self.num_modes = model.num_modes
        self.mode_interval = self.radius / self.num_modes
        self.time_decay = time_decay
        self.time_norm = time_norm

        if use_collision_loss:
            self.collision_loss = ESDFCollisionLoss()
        self.mul_ade = mulADE(k=1, 
                              with_grad=True,
                              mul_ade_loss=mul_ade_loss, 
                              max_horizon=max_horizon, 
                              learning_output=learning_output,
                              wtd_with_history=wtd_with_history,
                              wavelet=wavelet,
                              approximation_norm=approximation_norm,
                              use_dwt=use_dwt,
                              ).to(self.device)
        self.dynamic_weight = dynamic_weight
        print(f"self.device: {self.device}")
        init_weights = [float(w) for w in init_weights]
        self.weights = torch.tensor(init_weights, dtype=torch.float32)
        self.weights = self.weights.to(self.device)
        print(f"self.weights dtype after to device: {self.weights.dtype}")
        if self.dynamic_weight:
            # self.weights = torch.autograd.Variable(self.weights, requires_grad=True)
            self.weights.requires_grad = True
        self.mvn_loss = MVNLoss(k=3, with_grad=True).to(self.device)
        print('WARNING: Overall future time horizon is set to 80')
        self.OT = 80

    def on_fit_start(self) -> None:
        metrics_collection = MetricCollection(
            [
                minADE().to(self.device),
                minFDE().to(self.device),
                MR(miss_threshold=2).to(self.device),
                PredAvgADE().to(self.device),
                PredAvgFDE().to(self.device),
            ]
        )
        self.metrics = {
            "train": metrics_collection.clone(prefix="train/"),
            "val": metrics_collection.clone(prefix="val/"),
        }

    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        features, targets, scenarios = batch
        res = self.forward(features["feature"].data)

        losses = self._compute_objectives(res, features["feature"].data)
        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)

        return losses["loss"] if self.training else 0.0

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        bs, _, T, _ = res["prediction"].shape

        if self.use_contrast_loss:
            train_num = (bs // 3) * 2 if self.training else bs
        else:
            train_num = bs

        trajectory, probability, prediction = (
            res["trajectory"][:train_num],
            res["probability"][:train_num],
            res["prediction"][:train_num],
        )
        ref_free_trajectory = res.get("ref_free_trajectory", None)
        end = -self.OT+T if T < self.OT else None
        targets_pos = data["agent"]["target"][:train_num, :, -self.OT:end]
        valid_mask = data["agent"]["valid_mask"][:train_num, :, -self.OT:end]
        targets_vel = data["agent"]["velocity"][:train_num, :, -self.OT:end]
        target = torch.cat(
            [
                targets_pos[..., :2],
                torch.stack(
                    [targets_pos[..., 2].cos(), targets_pos[..., 2].sin()], dim=-1
                ),
                targets_vel,
            ],
            dim=-1,
        )

        # planning loss
        ego_reg_loss, ego_cls_loss, collision_loss = self.get_planning_loss(
            data, trajectory, probability, valid_mask[:, 0], target[:, 0], train_num
        )
        if ref_free_trajectory is not None:
            ego_ref_free_reg_loss = F.smooth_l1_loss(
                ref_free_trajectory[:train_num],
                target[:, 0, :, : ref_free_trajectory.shape[-1]],
                reduction="none",
            ).sum(-1)
            ego_ref_free_reg_loss = (
                ego_ref_free_reg_loss * valid_mask[:, 0]
            ).sum() / valid_mask[:, 0].sum()
        else:
            ego_ref_free_reg_loss = ego_reg_loss.new_zeros(1)

        # prediction loss
        prediction_loss = self.get_prediction_loss(
            data, prediction, valid_mask[:, 1:], target[:, 1:]
        ) if 'mvn' not in res.keys() else self.mvn_loss(res, data)

        if self.training and self.use_contrast_loss:
            contrastive_loss = self._compute_contrastive_loss(
                res["hidden"], data["data_n_valid_mask"]
            )
        else:
            contrastive_loss = prediction_loss.new_zeros(1)

        scope_loss = self.mul_ade(res, data)

        loss = (
            ego_reg_loss*self.weights[0] # 回归损失（路径近似程度）
            + ego_cls_loss*self.weights[1] # 分类损失（体现决策近似程度）
            + prediction_loss*self.weights[2] #运动预测损失
            + contrastive_loss # 三元组对比损失,为了学习一个良好的嵌入空间，使得来自同一类别或同一正样本对的向量更相似，而来自不同类别或负样本对的向量更不相似
            + collision_loss*self.weights[3] # 碰撞风险损失
            + ego_ref_free_reg_loss*self.weights[4] # 无参考回归损失，使用数据集GT作为参考Target
            + scope_loss*self.weights[5] # 路径细节损失
        )
        if self.training and self.dynamic_weight:
            self.losses = [ego_reg_loss, 
                           ego_cls_loss, 
                           prediction_loss, 
                            # contrastive_loss[0], 
                            collision_loss, 
                            ego_ref_free_reg_loss, 
                            scope_loss]
            loss = self.mgda_find_scaler(self.losses)

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss.item(),
            "cls_loss": ego_cls_loss.item(),
            "prediction_loss": prediction_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "collision_loss": collision_loss.item(),
            "ref_free_reg_loss": ego_ref_free_reg_loss.item(),
            "scope_loss": scope_loss.item(),
            "alpha_reg_loss": self.weights[0],
            "alpha_cls_loss": self.weights[1],
            "alpha_prediction_loss": self.weights[2],
            # "alpha_contrastive_loss": self.weights[5],
            "alpha_collision_loss": self.weights[3],
            "alpha_ref_free_reg_loss": self.weights[4],
            "alpha_scope_loss": self.weights[5],
        }

    def get_prediction_loss(self, data, prediction, valid_mask, target):
        """
        prediction: (bs, A-1, T, 6)
        valid_mask: (bs, A-1, T)
        target: (bs, A-1, 6)
        """
        prediction_loss = F.smooth_l1_loss(
            prediction[valid_mask], target[valid_mask], reduction="none"
        ).sum(-1)
        prediction_loss = prediction_loss.sum() / valid_mask.sum()

        return prediction_loss

    def get_planning_loss(self, data, trajectory, probability, valid_mask, target, bs):
        """
        trajectory: (bs, R, M, T, 4)
        valid_mask: (bs, T)
        """
        num_valid_points = valid_mask.sum(-1)
        endpoint_index = (num_valid_points / 10).long().clamp_(min=0, max=7)  # max 8s
        r_padding_mask = ~data["reference_line"]["valid_mask"][:bs].any(-1)  # (bs, R)
        future_projection = data["reference_line"]["future_projection"][:bs][
            torch.arange(bs), :, endpoint_index
        ]

        target_r_index = torch.argmin(
            future_projection[..., 1] + 1e6 * r_padding_mask, dim=-1
        )
        target_m_index = (
            future_projection[torch.arange(bs), target_r_index, 0] / self.mode_interval
        ).long()
        target_m_index.clamp_(min=0, max=self.num_modes - 1)

        target_label = torch.zeros_like(probability)
        target_label[torch.arange(bs), target_r_index, target_m_index] = 1

        best_trajectory = trajectory[torch.arange(bs), target_r_index, target_m_index]

        if self.use_collision_loss:
            collision_loss = self.collision_loss(
                best_trajectory, data["cost_maps"][:bs, :, :, 0].float()
            )
        else:
            collision_loss = trajectory.new_zeros(1)

        reg_loss = F.smooth_l1_loss(best_trajectory, target, reduction="none").sum(-1)
        if self.time_decay:
            # decay_weights = torch.exp(-(0.1*torch.arange(reg_loss.shape[-1], device=reg_loss.device))**2).unsqueeze(0)
            decay_weights = torch.exp(-0.1*torch.arange(reg_loss.shape[-1], device=reg_loss.device) 
                                        / torch.exp(torch.tensor(1, device=reg_loss.device))).unsqueeze(0)
            reg_loss = decay_weights*reg_loss/decay_weights.mean()
        if self.time_norm:
            reg_loss = reg_loss/(reg_loss.mean(dim=0, keepdim=True)+1e-6).clone().detach()
        reg_loss = (reg_loss * valid_mask).sum() / valid_mask.sum()

        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

        cls_loss = F.cross_entropy(
            probability.reshape(bs, -1), target_label.reshape(bs, -1).detach()
        )

        if self.regulate_yaw:
            heading_vec_norm = torch.norm(best_trajectory[..., 2:4], dim=-1)
            yaw_regularization_loss = F.l1_loss(
                heading_vec_norm, heading_vec_norm.new_ones(heading_vec_norm.shape)
            )
            reg_loss += yaw_regularization_loss

        return reg_loss, cls_loss, collision_loss

    def _compute_contrastive_loss(
        self, hidden, valid_mask, normalize=True, tempreture=0.1
    ):
        """
        Compute triplet loss

        Args:
            hidden: (3*bs, D)
        """
        if normalize:
            hidden = F.normalize(hidden, dim=1, p=2)

        if not valid_mask.any():
            return hidden.new_zeros(1)

        x_a, x_p, x_n = hidden.chunk(3, dim=0)

        x_a = x_a[valid_mask]
        x_p = x_p[valid_mask]
        x_n = x_n[valid_mask]

        logits_ap = (x_a * x_p).sum(dim=1) / tempreture
        logits_an = (x_a * x_n).sum(dim=1) / tempreture
        labels = x_a.new_zeros(x_a.size(0)).long()

        triplet_contrastive_loss = F.cross_entropy(
            torch.stack([logits_ap, logits_an], dim=1), labels
        )
        return triplet_contrastive_loss

    def _compute_metrics(self, res, data, prefix) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.

        :param predictions: model's predictions
        :param targets: ground truth targets
        :return: dictionary of metrics names and values
        """
        # get top 6 modes
        trajectory, probability = res["trajectory"], res["probability"]
        r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

        bs, R, M, T, _ = trajectory.shape
        trajectory = trajectory.reshape(bs, R * M, T, -1)
        probability = probability.reshape(bs, R * M)
        top_k_prob, top_k_index = probability.topk(6, dim=-1)
        top_k_traj = trajectory[torch.arange(bs)[:, None], top_k_index]

        outputs = {
            "trajectory": top_k_traj[..., :2],
            "probability": top_k_prob,
            "prediction": res["prediction"][..., :2],
            "prediction_target": data["agent"]["target"][:, 1:],
            "valid_mask": data["agent"]["valid_mask"][:, 1:, self.history_steps :],
        }
        target = data["agent"]["target"][:, 0]

        metrics = self.metrics[prefix](outputs, target)
        return metrics

    def _log_step(
        self,
        loss,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param metrics: dictionary of metrics names and values
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        self.log(
            f"loss/{prefix}_{loss_name}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True if prefix == "train" else False,
        )

        for key, value in objectives.items():
            self.log(
                f"objectives/{prefix}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if metrics is not None:
            self.log_dict(
                metrics,
                prog_bar=(prefix == "val"),
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def training_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "val")

    def test_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        # print("features:",features)

        data = [
            features['agent']['position'],  #0
            features['agent']['heading'],#1
            features['agent']['velocity'],#2
            features['agent']['shape'],#3
            features['agent']['category'],#4
            features['agent']['valid_mask'],#5

            features['map']['point_position'],#6
            features['map']['point_vector'],#7
            features['map']['point_orientation'],#8
            features['map']['polygon_center'],#9
            features['map']['polygon_type'],#10
            features['map']['polygon_on_route'],#11
            features['map']['polygon_tl_status'],#12
            features['map']['polygon_has_speed_limit'],#13
            features['map']['polygon_speed_limit'],#14
            features['map']['valid_mask'],#15

            features['reference_line']['position'],#16
            features['reference_line']['vector'],#17
            features['reference_line']['orientation'],#18
            features['reference_line']['valid_mask'],#19

            features['static_objects']['position'],#20
            features['static_objects']['heading'],#21
            features['static_objects']['shape'],#22
            features['static_objects']['category'],#23
            features['static_objects']['valid_mask'],#24

            features['current_state'],#25
        ]
        return self.model(data)
        # return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        # Get optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        # Get lr_scheduler
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]

    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     sh_layer = self.model.encoder_blocks[-1].mlp.fc2
    #     norms = grad_norm(sh_layer, norm_type=2)
    #     self.log_dict(norms)

    def mgda_find_scaler(self, losses, skip=5):
        if self.global_step%skip!=0:
            return torch.stack([l*w for l,w in zip(losses, self.weights)]).sum()
        sh_layer = self.model.encoder_blocks[-1].mlp.fc2
        gw = []
        for i in range(len(losses)):
            dl = torch.autograd.grad(losses[i], sh_layer.parameters(), retain_graph=True, create_graph=True, allow_unused=True)[0]
            # dl = torch.norm(dl)
            gw.append([dl])
        sol, min_norm = MinNormSolver.find_min_norm_element(gw)
        self.weights = sol
        weighted_loss = torch.stack([l*w for l,w in zip(losses, sol)]).sum()
        return weighted_loss