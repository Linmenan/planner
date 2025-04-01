import torch
import torch.nn as nn
import torch.nn.functional as F
from natten import NeighborhoodAttention1D
from timm.models.layers import DropPath
import math
class CustomizedNeighborhoodAttention1D_MH(nn.Module):
    def __init__(self, radius, dim, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        """
        多头邻域注意力模块。
        参数：
            radius: int，邻域半径（窗口大小 = 2*radius + 1）
            dim: int，输入特征总维度
            num_heads: int，多头数，要求 dim % num_heads == 0
            qkv_bias: bool，是否有偏置
            attn_drop, proj_drop: dropout 概率
        """
        super(CustomizedNeighborhoodAttention1D_MH, self).__init__()
        self.radius = radius
        self.window_size = 2 * radius + 1
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        参数：
            x: Tensor, shape = (B, L, dim)
        返回：
            out: Tensor, shape = (B, L, dim)
        """
        B, L, _ = x.shape
        # 生成 q, k, v: (B, L, 3*dim)
        qkv = self.qkv(x)
        # 重塑为 (B, L, 3, num_heads, head_dim) 再 permute 得到 (3, B, num_heads, L, head_dim)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个的形状为 (B, num_heads, L, head_dim)
        # 合并 B 和 num_heads → (B', L, head_dim)，其中 B' = B * num_heads
        q = q.reshape(B * self.num_heads, L, self.head_dim)
        k = k.reshape(B * self.num_heads, L, self.head_dim)
        v = v.reshape(B * self.num_heads, L, self.head_dim)
        
        # 对 k 和 v 沿序列维度填充，使得每个位置都可以构造一个完整的局部窗口
        # 将 k, v 转置为 (B', head_dim, L)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        # 填充最后一维（原来的 L 维），pad=(left, right)
        k_padded = F.pad(k_t, pad=(self.radius, self.radius), mode='constant', value=0)
        v_padded = F.pad(v_t, pad=(self.radius, self.radius), mode='constant', value=0)
        # 转回来 (B', L+2*radius, head_dim)
        k_padded = k_padded.transpose(1, 2)
        v_padded = v_padded.transpose(1, 2)
        
        # 使用基础算子替代 unfold：
        # k_padded: (B', L+2*radius, head_dim)
        Bprime = k_padded.shape[0]
        device = k_padded.device
        # 生成索引矩阵，形状 (L, window_size)；第 i 行为 [i, i+1, ..., i+window_size-1]
        idx = torch.arange(L, device=device).unsqueeze(1) + torch.arange(self.window_size, device=device).unsqueeze(0)
        # idx: (L, window_size)；扩展到 (Bprime, L, window_size)
        idx = idx.unsqueeze(0).expand(Bprime, L, self.window_size)
        # 将 idx 展开为 (Bprime, L * window_size)
        idx_flat = idx.reshape(Bprime, L * self.window_size)
        # 构造与 k_padded 在维度1一致的索引张量，k_padded 的形状为 (Bprime, L+2*radius, head_dim)
        # 扩展 idx_flat 到 (Bprime, L * window_size, head_dim)
        idx_expanded = idx_flat.unsqueeze(-1).expand(Bprime, L * self.window_size, self.head_dim)
        # 使用 gather 在维度1上提取对应的局部窗口数据，结果形状 (Bprime, L * window_size, head_dim)
        k_windows_flat = torch.gather(k_padded, dim=1, index=idx_expanded)
        v_windows_flat = torch.gather(v_padded, dim=1, index=idx_expanded)
        # reshape 成 (Bprime, L, window_size, head_dim)
        k_windows = k_windows_flat.reshape(Bprime, L, self.window_size, self.head_dim)
        v_windows = v_windows_flat.reshape(Bprime, L, self.window_size, self.head_dim)
        
        # 替代 einsum 计算 scores：
        # q: (B', L, head_dim) → unsqueeze(dim=2) → (B', L, 1, head_dim)
        # k_windows: (B', L, window_size, head_dim) → transpose(-2, -1) → (B', L, head_dim, window_size)
        scores = torch.matmul(q.unsqueeze(2), k_windows.transpose(-2, -1)).squeeze(2) / self.scale  # (B', L, window_size)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        
        # 替代 einsum 计算输出：out = sum_{w} attn * v_windows
        # attn: (B', L, window_size) → unsqueeze(dim=2): (B', L, 1, window_size)
        # v_windows: (B', L, window_size, head_dim)
        out = torch.matmul(attn.unsqueeze(2), v_windows).squeeze(2)  # (B', L, head_dim)
        
        # 恢复形状：(B, num_heads, L, head_dim)
        out = out.reshape(B, self.num_heads, L, self.head_dim)
        # 合并头 (B, L, dim)
        out = out.transpose(1, 2).reshape(B, L, self.dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class NATSequenceEncoder(nn.Module):
    def __init__(
        self,
        in_chans=3,
        embed_dim=32,
        mlp_ratio=3,
        kernel_size=[3, 3, 5],
        depths=[2, 2, 2],
        num_heads=[2, 4, 8],
        out_indices=[0, 1, 2],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.embed = ConvTokenizer(in_chans, embed_dim)
        self.num_levels = len(depths)
        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_levels)]
        self.out_indices = out_indices

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size[i],
                dilations=None,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
            )
            self.levels.append(level)

        for i_layer in self.out_indices:
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        n = self.num_features[-1]
        self.lateral_convs = nn.ModuleList()
        for i_layer in self.out_indices:
            self.lateral_convs.append(
                nn.Conv1d(self.num_features[i_layer], n, 3, padding=1)
            )

        self.fpn_conv = nn.Conv1d(n, n, 3, padding=1)

    def forward(self, x):
        """x: [B, C, T]"""
        x = self.embed(x)

        out = []
        for idx, level in enumerate(self.levels):
            x, xo = level(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(xo)
                out.append(x_out.permute(0, 2, 1).contiguous())

        laterals = [
            lateral_conv(out[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # print("laterals = ",laterals)
        for i in range(len(out) - 1, 0, -1):
            # print("i = ",i)
            # print("size(laterals[i]) = ",laterals[i].size())
            # print("size(laterals[i-1]) = ",laterals[i - 1].size())
            # print("laterals[i].shape[-1] = ",laterals[i].shape[-1])
            # print("laterals[i-1].shape[-1] = ",laterals[i-1].shape[-1])
            # print("factor = ",(laterals[i - 1].shape[-1] / laterals[i].shape[-1]))
            # print("float factor = ",float(laterals[i - 1].shape[-1] / laterals[i].shape[-1]))
            target_size = laterals[i - 1].size(-1)
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=target_size,  # 动态计算目标尺寸，保留在计算图中
                # scale_factor=float(laterals[i - 1].shape[-1] / laterals[i].shape[-1]),
                mode="linear",
                align_corners=False,
            )
            

        out = self.fpn_conv(laterals[0])

        return out[:, :, -1]


class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=32, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        # print("in_chans = ",in_chans)
        # print("embed_dim = ",embed_dim)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # print("forward forward forward forward forward forward")
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!x.type",type(x))
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!x(num).type",type(x[0,0,0]))
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!x",x)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!x.size()",x.size())
        # print("1111111111111111111111111111111111111111111111")
        x = self.proj(x).permute(0, 2, 1)  # B, C, L -> B, L, C
        # print("2222222222222222222222222222222222222222222222")
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv1d(
            dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
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
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NATLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        print(f"dim = {dim}")
        print(f"kernel_size = {kernel_size}")
        print(f"dilation = {dilation}")
        print(f"num_heads = {num_heads}")
        print(f"qkv_bias = {qkv_bias}")
        print(f"qk_scale = {qk_scale}")
        print(f"attn_drop = {attn_drop}")
        print(f"proj_drop = {drop}")
        # self.attn = NeighborhoodAttention1D(
        #     dim,
        #     kernel_size=kernel_size,
        #     dilation=dilation,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     attn_drop=attn_drop,
        #     proj_drop=drop,
        # )
        
        # self.attn = nn.MultiheadAttention(
        #     embed_dim=dim,       # 模型嵌入维度
        #     num_heads=num_heads, # 注意力头数
        #     dropout=attn_drop,   # 注意力 dropout 概率
        #     bias=qkv_bias        # 是否使用偏置项
        # )
        radius=(kernel_size-1)//2
        print(f"radius = {radius}")
        self.attn = CustomizedNeighborhoodAttention1D_MH(
            dim=dim,
            radius=radius,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )


        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
        
    # def forward(self, x):
    #     shortcut = x
    #     # 归一化
    #     x = self.norm1(x)  # (B, L, C)
    #     # 转换为 (L, B, C)
    #     x = x.transpose(0, 1)
    #     # 使用 MultiheadAttention 执行自注意力计算
    #     # 此处 attn 返回 (attn_output, attn_weights)，我们只关心输出 attn_output
    #     attn_output, _ = self.attn(x, x, x)
    #     # 恢复为 (B, L, C)
    #     attn_output = attn_output.transpose(0, 1)
    #     # 残差连接
    #     x = shortcut + self.drop_path(attn_output)
    #     # MLP 部分（带归一化及残差连接）
    #     x = x + self.drop_path(self.mlp(self.norm2(x)))
    #     return x

class NATBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        kernel_size,
        dilations=None,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                NATLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x, x
        return self.downsample(x), x


class PointsEncoder(nn.Module):
    def __init__(self, feat_channel, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_mlp = nn.Sequential(
            nn.Linear(feat_channel, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )
        self.second_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.encoder_channel),
        )

    # def forward(self, x, mask=None):
    #     """
    #     x : B M 3
    #     mask: B M
    #     -----------------
    #     feature_global : B C
    #     """

    #     bs, n, _ = x.shape
    #     device = x.device
    #     print(f"x.shape = {x.shape}, mask.shape = {mask.shape}")
    #     print(f"mask = {mask}")
    #     print(f"x[mask].shape = {x[mask].shape}, x[mask].type = {type(x[mask])}")
    #     x_valid = self.first_mlp(x[mask])  # B n 256
    #     x_features = torch.zeros(bs, n, 256, device=device)
    #     x_features[mask] = x_valid

    #     pooled_feature = x_features.max(dim=1)[0]
    #     x_features = torch.cat(
    #         [x_features, pooled_feature.unsqueeze(1).repeat(1, n, 1)], dim=-1
    #     )

    #     x_features_valid = self.second_mlp(x_features[mask])
    #     res = torch.zeros(bs, n, self.encoder_channel, device=device)
    #     res[mask] = x_features_valid

    #     res = res.max(dim=1)[0]
    #     print(f"res.shape = {res.shape}")
    #     return res
    def forward(self, x, mask=None):
        """
        x : B x M x 3
        mask: B x M (bool)
        -----------------
        feature_global : B x C
        """
        bs, n, _ = x.shape
        device = x.device
        # print(f"x.shape = {x.shape}, mask.shape = {mask.shape}")
        # print(f"mask = {mask}")

        # 将 x 和 mask 在前两个维度上拉平，便于使用索引
        x_flat = x.view(-1, x.shape[-1])         # [B*M, 3]
        mask_flat = mask.view(-1)                 # [B*M]

        # 用 nonzero 得到有效索引，确保后续索引操作形状明确
        valid_idx = mask_flat.nonzero(as_tuple=False).squeeze(1)  # [num_valid]

        # 获取有效点 y
        y = x_flat[valid_idx]
        # print(f"y.shape = {y.shape}")

        # 使用有效索引进行第一阶段 MLP 处理
        x_valid = self.first_mlp(x_flat[valid_idx])  # [num_valid, 256]
        
        # 构造一个全零的特征张量，再使用 scatter 写入有效点特征
        x_features = torch.zeros(bs * n, 256, device=device)
        # valid_idx.unsqueeze(1) 的形状为 [num_valid, 1]，expand 后为 [num_valid, 256]
        x_features = x_features.scatter(0, valid_idx.unsqueeze(1).expand(-1, 256), x_valid)
        # 恢复形状为 [B, M, 256]
        x_features = x_features.view(bs, n, 256)
        
        # 对每个 batch 取最大池化，得到 pooled_feature [B, 256]
        pooled_feature = x_features.max(dim=1)[0]
        # 将 pooled_feature 扩展为 [B, M, 256] 后与 x_features 在最后一维拼接，结果 [B, M, 512]
        x_features = torch.cat(
            [x_features, pooled_feature.unsqueeze(1).repeat(1, n, 1)], dim=-1
        )
        x_features_flat = x_features.view(-1, x_features.shape[-1])  # [B*M, 512]
        
        # 注意：这里使用 valid_idx 来索引而不是直接用布尔 mask
        x_features_valid = self.second_mlp(x_features_flat[valid_idx])  # [num_valid, encoder_channel]
        # print(f"x_features_valid.shape = {x_features_valid.shape}")

        # 构造结果张量
        res = torch.zeros(bs * n, self.encoder_channel, device=device)
        # 用 scatter 将 x_features_valid 按照 valid_idx 写入 res
        res = res.scatter(0, valid_idx.unsqueeze(1).expand(-1, self.encoder_channel), x_features_valid)
        res = res.view(bs, n, self.encoder_channel)
        # print(f"res.shape = {res.shape}")

        # 对每个 batch 在 M 维度上取最大池化
        res = res.max(dim=1)[0]
        # print(f"res.shape = {res.shape}")
        return res