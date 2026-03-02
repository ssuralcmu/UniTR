import torch
from torch import nn


class ConvFuser(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
            )

        self.enable_context_gate = self.model_cfg.get('ENABLE_CONTEXT_GATE', False)
        self.use_fixed_fusion_weight = self.model_cfg.get('USE_FIXED_FUSION_WEIGHT', False)
        self.fixed_img_weight = float(self.model_cfg.get('FIXED_FUSION_WEIGHT_IMG', 1.0))
        self.fixed_lidar_weight = float(self.model_cfg.get('FIXED_FUSION_WEIGHT_LIDAR', 1.0))
        if self.enable_context_gate:
            hidden_dim = self.model_cfg.get('CONTEXT_HIDDEN_DIM', 16)
            context_dim = self.model_cfg.get('CONTEXT_DIM', 2)
            self.context_gate = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 2)
            )
            nn.init.zeros_(self.context_gate[-1].weight)
            nn.init.zeros_(self.context_gate[-1].bias)

    def forward(self,batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        img_bev = batch_dict['spatial_features_img']
        lidar_bev = batch_dict['spatial_features']

        if self.use_fixed_fusion_weight:
            img_weight = img_bev.new_full((img_bev.shape[0], 1, 1, 1), self.fixed_img_weight)
            lidar_weight = lidar_bev.new_full((lidar_bev.shape[0], 1, 1, 1), self.fixed_lidar_weight)
            img_bev = img_bev * img_weight
            lidar_bev = lidar_bev * lidar_weight
            batch_dict['context_fusion_weight_img'] = img_weight.squeeze(-1).squeeze(-1)
            batch_dict['context_fusion_weight_lidar'] = lidar_weight.squeeze(-1).squeeze(-1)
        elif self.enable_context_gate and ('context_nr' in batch_dict):
            context_nr = batch_dict['context_nr'].to(img_bev.device).float()
            gate_logits = self.context_gate(context_nr)
            gate = 2.0 * torch.sigmoid(gate_logits)
            img_weight = gate[:, 0].view(-1, 1, 1, 1)
            lidar_weight = gate[:, 1].view(-1, 1, 1, 1)
            img_bev = img_bev * img_weight
            lidar_bev = lidar_bev * lidar_weight
            batch_dict['context_fusion_weight_img'] = img_weight.squeeze(-1).squeeze(-1)
            batch_dict['context_fusion_weight_lidar'] = lidar_weight.squeeze(-1).squeeze(-1)

        cat_bev = torch.cat([img_bev,lidar_bev],dim=1)
        mm_bev = self.conv(cat_bev)
        batch_dict['spatial_features'] = mm_bev
        return batch_dict