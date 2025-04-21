import math
import torch
import torch.nn as nn

from ..layers.fourier_embedding import FourierEmbedding


class StaticObjectsEncoder(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.obj_encoder = FourierEmbedding(2, dim, 64)
        self.type_emb = nn.Embedding(4, dim)

        nn.init.normal_(self.type_emb.weight, mean=0.0, std=0.01)

    # def forward(self, data):
    #     # pos = data["static_objects"]["position"]
    #     # heading = data["static_objects"]["heading"]
    #     # shape = data["static_objects"]["shape"]
    #     # category = data["static_objects"]["category"].long()
    #     # valid_mask = data["static_objects"]["valid_mask"]  # [bs, N]

    #     pos = data[26]
    #     heading = data[27]
    #     shape = data[28]
    #     category = data[29].long()
    #     valid_mask = data[30]  # [bs, N]

    #     obj_emb_tmp = self.obj_encoder(shape) + self.type_emb(category.long())
    #     obj_emb = torch.zeros_like(obj_emb_tmp)
    #     obj_emb[valid_mask] = obj_emb_tmp[valid_mask]

    #     heading = (heading + math.pi) % (2 * math.pi) - math.pi
    #     obj_pos = torch.cat([pos, heading.unsqueeze(-1)], dim=-1)

    #     return obj_emb, obj_pos, ~valid_mask
    def forward(self, data):
        # 从 data 中提取相关信息
        # pos: [B, N, 2]，heading: [B, N]，shape: 根据需要的形状，category: [B, N]，valid_mask: [B, N]
        pos = data[20]
        heading = data[21]
        shape = data[22]
        category = data[23].to(torch.int32)
        valid_mask = data[24]  # [B, N]

        # 计算静态物体的初步特征
        obj_emb_tmp = self.obj_encoder(shape) + self.type_emb(category)
        # obj_emb_tmp 的形状为 [B, N, dim]

        # 下面要将有效位置（valid_mask==True）的特征从 obj_emb_tmp 拷贝到一个全零张量中，
        # 避免直接布尔索引带来的问题，使用 scatter 替代
        _, N = valid_mask.shape
        bs, _, feat_dim = obj_emb_tmp.shape
        # print(f"obj_emb_tmp.shape:{obj_emb_tmp.shape}")
        # 对有效特征进行处理，但如果 N==0 或者有效障碍物数量为0时直接跳过 scatter 处理
        if N == 0:
            obj_emb = obj_emb_tmp  # 或者直接构造一个相同形状的空张量
        else:
            obj_emb = torch.where(valid_mask.unsqueeze(-1), obj_emb_tmp, torch.zeros_like(obj_emb_tmp))
            # # 创建一个全零张量，与 obj_emb_tmp 形状相同
            # obj_emb = torch.zeros_like(obj_emb_tmp)

            # # 将张量拉平处理，方便后续 scatter 操作
            # obj_emb_flat = obj_emb.view(-1, feat_dim)         # [B*N, feat_dim]
            # obj_emb_tmp_flat = obj_emb_tmp.view(-1, feat_dim)   # [B*N, feat_dim]
            # valid_mask_flat = valid_mask.view(-1)               # [B*N]

            # # 获取 valid_mask_flat 中 True 的位置索引
            # valid_idx = valid_mask_flat.nonzero(as_tuple=False).squeeze(1)  # [num_valid]

            # # 使用 scatter 将有效特征写入全零张量中：
            # # valid_idx.unsqueeze(1).expand(-1, feat_dim) 的形状为 [num_valid, feat_dim],
            # # 与 obj_emb_tmp_flat[valid_idx] 的形状一致
            # obj_emb_flat = obj_emb_flat.scatter(0,
            #                                     valid_idx.unsqueeze(1).expand(-1, feat_dim),
            #                                     obj_emb_tmp_flat[valid_idx])
            # # 将更新后的 obj_emb_flat 恢复为原始形状 [B, N, feat_dim]
            # obj_emb = obj_emb_flat.view(bs, N, feat_dim)

            # 对 heading 进行归一化
            heading = (heading + math.pi) % (2 * math.pi) - math.pi
            # 拼接物体位置和 heading（扩展 heading 的最后一维）
        
        obj_pos = torch.cat([pos, heading.unsqueeze(-1)], dim=-1)

        # 返回特征、物体位置信息以及无效掩码（取反 valid_mask）
        return obj_emb, obj_pos, ~valid_mask
