from copy import deepcopy
import math

import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.feature_builders.pluto_feature_builder import PlutoFeatureBuilder

from .layers.fourier_embedding import FourierEmbedding
from .layers.transformer import TransformerEncoderLayer
from .modules.agent_encoder import AgentEncoder
from .modules.agent_predictor import AgentPredictor
from .modules.map_encoder import MapEncoder
from .modules.static_objects_encoder import StaticObjectsEncoder
# from .modules.planning_decoder import PlanningDecoder
from .modules.hierachical_decoder import HierachicalDecoder
from .layers.mlp_layer import MLPLayer

# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)

def print_structure(data, indent=0):
    """
    递归打印数据结构及其中张量的尺寸和数据类型。
    """
    prefix = " " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{prefix}{key}:")
            print_structure(value, indent=indent+4)
    elif isinstance(data, (list, tuple)):
        print(f"{prefix}{type(data).__name__} (length: {len(data)})")
        for idx, item in enumerate(data):
            print(f"{prefix}Index {idx}:")
            print_structure(item, indent=indent+4)
    elif torch.is_tensor(data):
        # 同时打印形状、数据类型和设备信息
        print(f"{prefix}Tensor, shape: {data.shape}, dtype: {data.dtype}, device: {data.device}")
    else:
        print(f"{prefix}{data} (type: {type(data)})")


class PlanningModel(TorchModuleWrapper):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        decoder_depth=4,
        drop_path=0.2,
        dropout=0.1,
        num_heads=8,
        num_modes=6,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        use_hidden_proj=False,
        cat_x=False,
        ref_free_traj=False,
        recursive_decoder:bool=False,
        residual_decoder:bool=False,
        multihead_decoder:bool=False,
        independent_detokenizer:bool=False,
        mvn_loss=False,
        wtd_with_history=False,
        feature_builder: PlutoFeatureBuilder = PlutoFeatureBuilder(),
        plot_debug = False,
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.plot_debug = plot_debug
        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.use_hidden_proj = use_hidden_proj
        self.num_modes = num_modes
        self.radius = feature_builder.radius
        self.ref_free_traj = ref_free_traj

        self.pos_emb = FourierEmbedding(3, dim, 64)

        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
            
        )

        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
            use_lane_boundary=True,
        )

        self.static_objects_encoder = StaticObjectsEncoder(dim=dim)

        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp,debug=self.plot_debug)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)

        self.agent_predictor = AgentPredictor(dim=dim, future_steps=future_steps)
        self.planning_decoder = HierachicalDecoder(
            num_mode=num_modes,
            decoder_depth=decoder_depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            dropout=dropout,
            future_steps=future_steps,
            history_steps=history_steps,
            cat_x=cat_x,
            recursive_decoder=recursive_decoder,
            residual_decoder=residual_decoder,
            multihead_decoder=multihead_decoder,
            wtd_with_history=wtd_with_history,
            independent_detokenizer=independent_detokenizer,
        )

        if use_hidden_proj:
            self.hidden_proj = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )

        if self.ref_free_traj:
            self.ref_free_decoder = MLPLayer(dim, 2 * dim, future_steps * 4)

        self.apply(self._init_weights)

        self.mvn_loss = mvn_loss
        if self.mvn_loss:
            self.mvn_decoder = MLPLayer(dim, 3 * dim, future_steps * 4)
    def set_debug(self, debug):
        self.plot_debug = debug

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def custom_atan2(self, rot_sine, rot_cosine):
        # 防止除数为 0
        safe_rot_cosine = torch.where(rot_cosine == 0, torch.full_like(rot_cosine, 1e-6), rot_cosine)
        # 计算基本的 arctan 值
        rot = torch.atan(rot_sine / safe_rot_cosine)
        # 对于第二象限：sine >= 0 且 cosine < 0，角度加 π
        mask_yp = (rot_sine >= 0) & (rot_cosine < 0)
        # 对于第三象限：sine < 0 且 cosine < 0，角度减 π
        mask_yn = (rot_sine < 0) & (rot_cosine < 0)
        # 根据掩码调整角度
        rot = torch.where(mask_yp, rot + torch.pi, rot)
        rot = torch.where(mask_yn, rot - torch.pi, rot)
        return rot

    def forward(self, data):
        # print("data )))))))))))))))))))))))",data[6])
        # print_structure(data)

        # agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        # agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        # agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        # polygon_center = data["map"]["polygon_center"]
        # polygon_mask = data["map"]["valid_mask"]
        agent_pos = data[0][:, :, self.history_steps - 1]
        agent_heading = data[1][:, :, self.history_steps - 1]
        agent_mask = data[5][:, :, : self.history_steps]
        polygon_center = data[9]
        polygon_mask = data[15]

        bs, A = agent_pos.shape[0:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)
        # print(f"agent_key_padding.shape: {agent_key_padding.shape}")
        # print(f"polygon_key_padding.shape: {polygon_key_padding.shape}")
        # print(f"key_padding_mask.shape: {key_padding_mask.shape}")

        x_agent = self.agent_encoder(data)
        # print(f"x_agent.shape: {x_agent.shape}")
        x_polygon = self.map_encoder(data)
        # print(f"x_polygon.shape: {x_polygon.shape}")
        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)#静态障碍物形状+位置编码、静态障碍物位置、无效掩码
        # print(f"x_static.shape: {x_static.shape}")

        x = torch.cat([x_agent, x_polygon, x_static], dim=1)
        vary_nums = [x_agent.shape[1], x_polygon.shape[1], x_static.shape[1]]
        pos = torch.cat([pos, static_pos], dim=1)
        pos_embed = self.pos_emb(pos)

        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)
        # print(f"static_key_padding.shape: {static_key_padding.shape}")
        # print(f"key_padding_mask.shape: {key_padding_mask.shape}")
        x = x + pos_embed
        # print(f"x.shape: {x.shape}")
        # print(f"key_padding_mask.shape: {key_padding_mask.shape}")
        # print(f"key_padding_mask: {key_padding_mask}")
        for index, blk in enumerate(self.encoder_blocks):
        # for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False, index=[index,len(self.encoder_blocks)],vary_nums=vary_nums)
        x = self.norm(x)
        prediction = self.agent_predictor(x[:, 1:A])

        # ref_line_available = data["reference_line"]["position"].shape[1] > 0
        ref_line_available = data[16].shape[1] > 0

        if ref_line_available:
            trajectory, probability, details = self.planning_decoder(
                data, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask}
                )
        else:
            trajectory, probability = None, None
            details = None

        # trajectory, probability, details = self.planning_decoder(
        #     data, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask})
        # trajectory, probability = self.planning_decoder(
        #     data, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask})
        out = {
            "trajectory": trajectory,  # out0 多模态轨迹（x,y,cos,sin,vx,vy）
            "probability": probability,  # out1 (bs, R, M) 多模态轨迹打分
            "prediction": prediction,  # out 2 (bs, A-1, T, 2) Agent运动预测（x,y,cos,sin,vx,vy） 
            "details": details, 
        }

        if self.use_hidden_proj:
            out["hidden"] = self.hidden_proj(x[:, 0])

        if self.ref_free_traj:
            ref_free_traj = self.ref_free_decoder(x[:, 0]).reshape(
                bs, self.future_steps, 4
            )
            out["ref_free_trajectory"] = ref_free_traj

        if self.mvn_loss:
            mvn = self.mvn_decoder(x[:, 1:]).reshape(bs, self.future_steps, 3)
            out["mvn"] = mvn

        if not self.training:
            if self.ref_free_traj:
                ref_free_traj_angle = torch.arctan2(
                    ref_free_traj[..., 3], ref_free_traj[..., 2]
                )
                ref_free_traj = torch.cat(
                    [ref_free_traj[..., :2], ref_free_traj_angle.unsqueeze(-1)], dim=-1
                )
                out["output_ref_free_trajectory"] = ref_free_traj

            output_prediction = torch.cat(
                [
                    prediction[..., :2] + agent_pos[:, 1:A, None],
                    self.custom_atan2(prediction[..., 3], prediction[..., 2]).unsqueeze(-1)
                    + agent_heading[:, 1:A, None, None],
                    prediction[..., 4:6],
                ],
                dim=-1,
            )
            out["output_prediction"] = output_prediction # out 3 Agent运动预测（x,y,yaw,vx,vy） 

            if trajectory is not None:
                # r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
                r_padding_mask = ~data[19].any(-1)
                probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

                angle = self.custom_atan2(trajectory[..., 3], trajectory[..., 2])
                out_trajectory = torch.cat(
                    [trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
                )

                bs, R, M, T, _ = out_trajectory.shape
                flattened_probability = probability.reshape(bs, R * M)
                best_trajectory = out_trajectory.reshape(bs, R * M, T, -1)[
                    torch.arange(bs), flattened_probability.argmax(-1)
                ]

                out["output_trajectory"] = best_trajectory # out 4 打分最高轨迹（x,y,yaw）
                out["candidate_trajectories"] = out_trajectory # out 5 多模态轨迹（x,y,yaw）
            else:
                # TODO (1,0,0) originally
                out["output_trajectory"] = out["output_ref_free_trajectory"]
                out["probability"] = torch.ones(1, 1, 1)
                out["candidate_trajectories"] = torch.zeros(
                    1, 1, 1, self.future_steps, 3
                )


        # for key,val in out.items():
        #     if key=="details":
        #         for i in val:
        #             print(f"i = {i}, value.shape = {type(i)}")
        #     else:
        #         print(f"key = {key}, value.shape = {type(val)}")

        return out
