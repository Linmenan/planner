import torch
import torch.nn as nn

from ..layers.embedding import PointsEncoder
from ..layers.fourier_embedding import FourierEmbedding


class MapEncoder(nn.Module):
    def __init__(
        self,
        polygon_channel=6,
        dim=128,
        use_lane_boundary=False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.use_lane_boundary = use_lane_boundary
        self.polygon_channel = (
            polygon_channel + 4 if use_lane_boundary else polygon_channel
        )

        self.polygon_encoder = PointsEncoder(self.polygon_channel, dim)
        self.speed_limit_emb = FourierEmbedding(1, dim, 64)

        self.type_emb = nn.Embedding(3, dim)
        self.on_route_emb = nn.Embedding(2, dim)
        self.traffic_light_emb = nn.Embedding(4, dim)
        self.unknown_speed_emb = nn.Embedding(1, dim)

    # def forward(self, data) -> torch.Tensor:
    #     # polygon_center = data["map"]["polygon_center"]
    #     # polygon_type = data["map"]["polygon_type"].long()
    #     # polygon_on_route = data["map"]["polygon_on_route"].long()
    #     # polygon_tl_status = data["map"]["polygon_tl_status"].long()
    #     # polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]
    #     # polygon_speed_limit = data["map"]["polygon_speed_limit"]
    #     # point_position = data["map"]["point_position"]
    #     # point_vector = data["map"]["point_vector"]
    #     # point_orientation = data["map"]["point_orientation"]
    #     # valid_mask = data["map"]["valid_mask"]

    #     polygon_center = data[11]
    #     polygon_type = data[14].long()
    #     polygon_on_route = data[15].long()
    #     polygon_tl_status = data[16].long()
    #     polygon_has_speed_limit = data[17]
    #     polygon_speed_limit = data[18]
    #     point_position = data[7]
    #     point_vector = data[8]
    #     point_orientation = data[9]
    #     valid_mask = data[20]
    #     if self.use_lane_boundary:
    #         polygon_feature = torch.cat(
    #             [
    #                 point_position[:, :, 0] - polygon_center[..., None, :2],
    #                 point_vector[:, :, 0],
    #                 torch.stack(
    #                     [
    #                         point_orientation[:, :, 0].cos(),
    #                         point_orientation[:, :, 0].sin(),
    #                     ],
    #                     dim=-1,
    #                 ),
    #                 point_position[:, :, 1] - point_position[:, :, 0],
    #                 point_position[:, :, 2] - point_position[:, :, 0],
    #             ],
    #             dim=-1,
    #         )
    #     else:
    #         polygon_feature = torch.cat(
    #             [
    #                 point_position[:, :, 0] - polygon_center[..., None, :2],
    #                 point_vector[:, :, 0],
    #                 torch.stack(
    #                     [
    #                         point_orientation[:, :, 0].cos(),
    #                         point_orientation[:, :, 0].sin(),
    #                     ],
    #                     dim=-1,
    #                 ),
    #             ],
    #             dim=-1,
    #         )

    #     bs, M, P, C = polygon_feature.shape
    #     valid_mask = valid_mask.view(bs * M, P)
    #     polygon_feature = polygon_feature.reshape(bs * M, P, C)

    #     x_polygon = self.polygon_encoder(polygon_feature, valid_mask).view(bs, M, -1)

    #     x_type = self.type_emb(polygon_type)
    #     x_on_route = self.on_route_emb(polygon_on_route)
    #     x_tl_status = self.traffic_light_emb(polygon_tl_status)
    #     x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device)
    #     x_speed_limit[polygon_has_speed_limit] = self.speed_limit_emb(
    #         polygon_speed_limit[polygon_has_speed_limit].unsqueeze(-1)
    #     )
    #     x_speed_limit[~polygon_has_speed_limit] = self.unknown_speed_emb.weight

    #     x_polygon += x_type + x_on_route + x_tl_status + x_speed_limit

    #     return x_polygon

    def forward(self, data) -> torch.Tensor:
        # 从 data 中提取各个字段（索引对应之前定义的顺序）
        polygon_center = data[9]
        polygon_type = data[10].to(torch.int32)
        polygon_on_route = data[11].to(torch.int32)
        polygon_tl_status = data[12].to(torch.int32)
        polygon_has_speed_limit = data[13]  # 布尔张量，形状 [B, M]
        polygon_speed_limit = data[14]        # 浮点数张量，形状 [B, M]
        point_position = data[6]
        point_vector = data[7]
        point_orientation = data[8]
        valid_mask = data[15]

        # 构造多边形特征
        if self.use_lane_boundary:
            polygon_feature = torch.cat(
                [
                    point_position[:, :, 0] - polygon_center[..., None, :2],
                    point_vector[:, :, 0],
                    torch.stack(
                        [
                            point_orientation[:, :, 0].cos(),
                            point_orientation[:, :, 0].sin(),
                        ],
                        dim=-1,
                    ),
                    point_position[:, :, 1] - point_position[:, :, 0],
                    point_position[:, :, 2] - point_position[:, :, 0],
                ],
                dim=-1,
            )
        else:
            polygon_feature = torch.cat(
                [
                    point_position[:, :, 0] - polygon_center[..., None, :2],
                    point_vector[:, :, 0],
                    torch.stack(
                        [
                            point_orientation[:, :, 0].cos(),
                            point_orientation[:, :, 0].sin(),
                        ],
                        dim=-1,
                    ),
                ],
                dim=-1,
            )

        # polygon_feature: [B, M, P, C]
        bs, M, P, C = polygon_feature.shape
        valid_mask = valid_mask.view(bs * M, P)
        polygon_feature = polygon_feature.reshape(bs * M, P, C)

        # 通过 polygon_encoder 得到多边形特征 x_polygon，形状为 [B, M, dim]
        x_polygon = self.polygon_encoder(polygon_feature, valid_mask).view(bs, M, -1)

        # 获取其他属性嵌入，形状均为 [B, M, dim]
        x_type = self.type_emb(polygon_type)
        x_on_route = self.on_route_emb(polygon_on_route)
        x_tl_status = self.traffic_light_emb(polygon_tl_status)

        # 构造速度限制特征
        x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device)
        # 为方便 scatter，将 x_speed_limit 展平
        x_speed_limit_flat = x_speed_limit.view(bs * M, self.dim)  # [B*M, dim]
        # polygon_has_speed_limit: [B, M] -> [B*M]
        has_speed_flat = polygon_has_speed_limit.view(-1)  
        
        # 对于有效（True）的位置，计算更新值
        valid_idx = has_speed_flat.nonzero(as_tuple=False).squeeze(1)  # [num_valid]
        # polygon_speed_limit: [B, M]，取出有效位置并 unsqueeze 变成 [num_valid, 1]
        speed_vals = polygon_speed_limit[polygon_has_speed_limit].unsqueeze(-1)
        # 经过速度嵌入后，得到更新值 shape 为 [num_valid, dim]
        update_speed = self.speed_limit_emb(speed_vals)
        # 使用 scatter 将 update_speed 写入 x_speed_limit_flat 的对应位置
        x_speed_limit_flat = x_speed_limit_flat.scatter(0,
                                                        valid_idx.unsqueeze(1).expand(-1, self.dim),
                                                        update_speed)
        
        # 对于无效（False）的索引
        invalid_idx = (~has_speed_flat).nonzero(as_tuple=False).squeeze(1)  # [num_invalid]
        # unknown_speed_emb.weight 的 shape 为 [1, dim]，扩展为 [num_invalid, dim]
        unknown_val = self.unknown_speed_emb.weight.expand(invalid_idx.shape[0], self.dim)
        x_speed_limit_flat = x_speed_limit_flat.scatter(0,
                                                        invalid_idx.unsqueeze(1).expand(-1, self.dim),
                                                        unknown_val)
        # 恢复 x_speed_limit 的形状为 [B, M, dim]
        x_speed_limit = x_speed_limit_flat.view(bs, M, self.dim)

        # 将各个特征相加
        x_polygon = x_polygon + x_type + x_on_route + x_tl_status + x_speed_limit

        return x_polygon
