from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch import Tensor


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        debug=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = torch.nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=qkv_bias,
            dropout=attn_drop,
            batch_first=True,
        )
        self.debug = debug
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        return_attn_weights=False,
        vary_nums = [],
        index=[],
    ):
        src2 = self.norm1(src)
        src2, attn = self.attn(
            query=src2,
            key=src2,
            value=src2,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )
        # print(f"attn = {attn}")
        if self.debug and index[0]+1==index[1]:
            import numpy as np
            import plotly.graph_objects as go
            # attn_mtx: shape [bs, A, A]，取第一个样本
            # print(f"attn.shape = {attn.shape}")
            # print(f"key_padding_mask.shape = {key_padding_mask.shape}")
            _,A = key_padding_mask.shape
            attn_data = attn[0].cpu().detach().numpy()                       # (A, A)
            mask_data = key_padding_mask[0].float().cpu().detach().numpy()   # (A,)
            
            # 设置下方水平掩码显示高度和右侧垂直掩码显示宽度（单位：数据坐标）
            mask_display_height = A//10  # 下方区域高度
            mask_display_width = A//10   # 右侧区域宽度

            # 构造水平掩码数据：将 1 行的掩码广播为 (mask_display_height, A)
            mask_horizontal = np.tile(mask_data[np.newaxis, :], (mask_display_height, 1))
            # 构造垂直掩码数据：将 1D 掩码转置后广播为 (A, mask_display_width)
            mask_vertical = np.tile(mask_data[:, np.newaxis], (1, mask_display_width))

            # 定义整个图像数据区域：
            # 热力图数据区域：x: [0, A], y: [mask_display_height, A+mask_display_height]
            # 水平掩码区域：x: [0, A], y: [A+mask_display_height, A+mask_display_height+mask_display_height]
            # 垂直掩码区域：x: [A, A+mask_display_width], y: [mask_display_height, A+mask_display_height]
            total_x = A + mask_display_width
            total_y = A + mask_display_height

            # 创建 Plotly 图形
            fig = go.Figure()

            # 添加 Attention 热力图 trace（注意：设置 extent 保证数据区域与坐标一致）
            x_vals = np.arange(A)
            y_vals = np.arange(A)
            fig.add_trace(go.Heatmap(
                z=attn_data,
                x=x_vals,
                y=y_vals,
                colorscale='Viridis',
                colorbar=dict(title="Attention", len=0.5, y=0.25)
            ))


            # 添加水平掩码 trace，显示在热力图下方
            x_vals_mask = np.arange(A)
            y_vals_mask = np.arange(A, A + mask_display_height)
            fig.add_trace(go.Heatmap(
                z=mask_horizontal,
                x=x_vals_mask,
                y=y_vals_mask,
                colorscale='Gray',
                showscale=False
            ))

            # 添加垂直掩码 trace，显示在热力图右侧
            x_vals_mask_v = np.arange(A, A + mask_display_width)
            y_vals_mask_v = np.arange(A)
            fig.add_trace(go.Heatmap(
                z=mask_vertical,
                x=x_vals_mask_v,
                y=y_vals_mask_v,
                colorscale='Gray',
                showscale=False
            ))

            # 添加边界划分线：根据 vary_nums 对热力图区域进行分割
            # 边界在 x 轴：x1 = num_agents, x2 = num_agents + num_map
            # 假设 vary_nums[0] 表示 agent 数量，vary_nums[1] 表示 map 数量
            x1 = vary_nums[0] - 0.5
            x2 = vary_nums[0] + vary_nums[1] - 0.5
            y1 = vary_nums[0] - 0.5
            y2 = vary_nums[0] + vary_nums[1] - 0.5

            # 添加垂直边界线
            fig.add_shape(type="line",
                        x0=x1, y0=-0.5, x1=x1, y1=A-0.5,
                        line=dict(color="red", width=2))
            fig.add_shape(type="line",
                        x0=x2, y0=-0.5, x1=x2, y1=A-0.5,
                        line=dict(color="red", width=2))
            # 添加水平边界线
            fig.add_shape(type="line",
                        x0=-0.5, y0=y1, x1=A-0.5, y1=y1,
                        line=dict(color="red", width=2))
            fig.add_shape(type="line",
                        x0=-0.5, y0=y2, x1=A-0.5, y1=y2,
                        line=dict(color="red", width=2))

            # 设置坐标轴，左上角为原点，x 向右，y 向下为正
            fig.update_layout(
                title="Attention Heatmap with Partitioned Regions",
                xaxis=dict(title="Key", side='bottom', tickmode='linear', dtick=1, range=[-0.5, A + mask_display_width-0.5]),
                yaxis=dict(title="Query", tickmode='linear', dtick=1, range=[A + mask_display_height-0.5, -0.5]),
                width=500,
                height=500
            )

            fig.show(renderer="notebook")


        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))

        if return_attn_weights:
            return src, attn

        return src


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4,
        qkv_bias=False,
        dropout=0.1,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_first=True,
    ):
        super().__init__()

        self.norm_first = norm_first
        self.attn = torch.nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=qkv_bias,
            dropout=attn_drop,
            batch_first=True,
        )

        self.linear1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.activation = act_layer()
        self.linear2 = nn.Linear(int(mlp_ratio * dim), dim)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x,
        memory,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        if self.norm_first:
            x = x + self._mha_block(self.norm1(x), memory, mask, key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._mha_block(x, memory, mask, key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        return self.dropout3(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
