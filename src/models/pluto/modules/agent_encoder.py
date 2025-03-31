import torch
import torch.nn as nn

from ..layers.common_layers import build_mlp
from ..layers.embedding import NATSequenceEncoder


class AgentEncoder(nn.Module):
    def __init__(
        self,
        state_channel=6,
        history_channel=9,
        dim=128,
        hist_steps=21,
        use_ego_history=False,
        drop_path=0.2,
        state_attn_encoder=True,
        state_dropout=0.75,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.state_channel = state_channel
        self.use_ego_history = use_ego_history
        self.hist_steps = hist_steps
        self.state_attn_encoder = state_attn_encoder

        self.history_encoder = NATSequenceEncoder(
            in_chans=history_channel, embed_dim=dim // 4, drop_path_rate=drop_path
        )

        if not use_ego_history:
            if not self.state_attn_encoder:
                self.ego_state_emb = build_mlp(state_channel, [dim] * 2, norm="bn")
            else:
                self.ego_state_emb = StateAttentionEncoder(
                    state_channel, dim, state_dropout
                )

        self.type_emb = nn.Embedding(4, dim)

    @staticmethod
    def to_vector(feat, valid_mask):
        vec_mask = valid_mask[..., :-1] & valid_mask[..., 1:]

        while len(vec_mask.shape) < len(feat.shape):
            vec_mask = vec_mask.unsqueeze(-1)

        return torch.where(
            vec_mask,
            feat[:, :, 1:, ...] - feat[:, :, :-1, ...],
            torch.zeros_like(feat[:, :, 1:, ...]),
        )

    # def forward(self, data):
    #     T = self.hist_steps

    #     # position = data["agent"]["position"][:, :, :T]
    #     # heading = data["agent"]["heading"][:, :, :T]
    #     # velocity = data["agent"]["velocity"][:, :, :T]
    #     # shape = data["agent"]["shape"][:, :, :T]
    #     # category = data["agent"]["category"].long()
    #     # valid_mask = data["agent"]["valid_mask"][:, :, :T]

    #     position = data[0][:, :, :T]
    #     heading = data[1][:, :, :T]
    #     velocity = data[2][:, :, :T]
    #     shape = data[3][:, :, :T]
    #     category = data[4].long()
    #     valid_mask = data[5][:, :, :T]

    #     heading_vec = self.to_vector(heading, valid_mask)
    #     valid_mask_vec = valid_mask[..., 1:] & valid_mask[..., :-1]
    #     agent_feature = torch.cat(
    #         [
    #             self.to_vector(position, valid_mask),
    #             self.to_vector(velocity, valid_mask),
    #             torch.stack([heading_vec.cos(), heading_vec.sin()], dim=-1),
    #             shape[:, :, 1:],
    #             valid_mask_vec.float().unsqueeze(-1),
    #         ],
    #         dim=-1,
    #     )
    #     bs, A, T, _ = agent_feature.shape
    #     agent_feature = agent_feature.view(bs * A, T, -1)
    #     valid_agent_mask = valid_mask.any(-1).flatten()

    #     x_agent_tmp = self.history_encoder(
    #         agent_feature[valid_agent_mask].permute(0, 2, 1).contiguous()
    #     )
    #     x_agent = torch.zeros(bs * A, self.dim, device=position.device)
    #     x_agent[valid_agent_mask] = x_agent_tmp
    #     x_agent = x_agent.view(bs, A, self.dim)

    #     if not self.use_ego_history:
    #         # ego_feature = data["current_state"][:, : self.state_channel]
    #         ego_feature = data[31][:, : self.state_channel]
    #         x_ego = self.ego_state_emb(ego_feature)
    #         x_agent[:, 0] = x_ego

    #     x_type = self.type_emb(category)

    #     return x_agent + x_type
    def forward(self, data):
        T = self.hist_steps

        # 提取 agent 数据（前 T 帧）
        position = data[0][:, :, :T]
        heading = data[1][:, :, :T]
        velocity = data[2][:, :, :T]
        shape = data[3][:, :, :T]
        category = data[4].to(torch.int32)
        valid_mask = data[5][:, :, :T]

        # 将 heading 转换为向量表示
        heading_vec = self.to_vector(heading, valid_mask)
        valid_mask_vec = valid_mask[..., 1:] & valid_mask[..., :-1]
        agent_feature = torch.cat(
            [
                self.to_vector(position, valid_mask),
                self.to_vector(velocity, valid_mask),
                torch.stack([heading_vec.cos(), heading_vec.sin()], dim=-1),
                shape[:, :, 1:],
                valid_mask_vec.float().unsqueeze(-1),
            ],
            dim=-1,
        )
        bs, A, T, _ = agent_feature.shape
        # 合并 batch 和 agent 维度，便于后续索引操作
        agent_feature = agent_feature.view(bs * A, T, -1)
        # valid_agent_mask: [bs*A]，表示每个 agent 是否至少有一个有效点
        valid_agent_mask = valid_mask.any(-1).flatten()  

        # 选出有效的 agent 特征，并做 transpose (通道维度放到中间)
        x_agent_tmp = self.history_encoder(
            agent_feature[valid_agent_mask].permute(0, 2, 1).contiguous()
        )
        # x_agent_tmp 的形状为 [num_valid, self.dim]

        # 构造全零张量 x_agent，形状 [bs*A, self.dim]
        x_agent = torch.zeros(bs * A, self.dim, device=position.device)
        # 替换直接赋值 x_agent[valid_agent_mask] = x_agent_tmp 为 scatter 操作：
        valid_idx = valid_agent_mask.nonzero(as_tuple=False).squeeze(1)  # [num_valid]
        # 将 x_agent_tmp 按照 valid_idx 写入 x_agent
        x_agent = x_agent.scatter(0, valid_idx.unsqueeze(1).expand(-1, self.dim), x_agent_tmp)
        # 恢复形状为 [bs, A, self.dim]
        x_agent = x_agent.view(bs, A, self.dim)

        # 对 ego 历史进行替换（如果不使用 ego 历史，则将第一 agent 置为当前状态）
        if not self.use_ego_history:
            ego_feature = data[25][:, : self.state_channel]
            x_ego = self.ego_state_emb(ego_feature)
            x_agent[:, 0] = x_ego

        # 处理类型嵌入
        x_type = self.type_emb(category)

        return x_agent + x_type


class StateAttentionEncoder(nn.Module):
    def __init__(self, state_channel, dim, state_dropout=0.5) -> None:
        super().__init__()

        self.state_channel = state_channel
        self.state_dropout = state_dropout
        self.linears = nn.ModuleList([nn.Linear(1, dim) for _ in range(state_channel)])
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.pos_embed = nn.Parameter(torch.Tensor(1, state_channel, dim))
        self.query = nn.Parameter(torch.Tensor(1, 1, dim))

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x):
        x_embed = []
        for i, linear in enumerate(self.linears):
            x_embed.append(linear(x[:, i, None]))
        x_embed = torch.stack(x_embed, dim=1)
        pos_embed = self.pos_embed.repeat(x_embed.shape[0], 1, 1)
        x_embed += pos_embed

        if self.training and self.state_dropout > 0:
            visible_tokens = torch.zeros(
                (x_embed.shape[0], 3), device=x.device, dtype=torch.bool
            )
            dropout_tokens = (
                torch.rand((x_embed.shape[0], self.state_channel - 3), device=x.device)
                < self.state_dropout
            )
            key_padding_mask = torch.concat([visible_tokens, dropout_tokens], dim=1)
        else:
            key_padding_mask = None

        query = self.query.repeat(x_embed.shape[0], 1, 1)

        x_state = self.attn(
            query=query,
            key=x_embed,
            value=x_embed,
            key_padding_mask=key_padding_mask,
        )[0]

        return x_state[:, 0]
