import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
import sys
sys.path.append("../utils")
sys.path.append("..")
sys.path.append("./")
# from checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
import random
from extensions.chamfer_dist import ChamferDistanceL2
# from pointnet2_utils import PointNetFeaturePropagation_
from pointnet2_utils import PointNetFeaturePropagation_
from utils.logger import *
# from modules import *
from .modules import *
# Hierarchical Encoder
from pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation,index_points

class H_Encoder_seg(nn.Module):

    def __init__(self, encoder_depths=[5, 5, 5], num_heads=6, encoder_dims=[96, 192, 384], local_radius=[0.32, 0.64, 1.28]):
        super().__init__()

        self.encoder_depths = encoder_depths
        self.encoder_num_heads = num_heads
        self.encoder_dims = encoder_dims
        self.local_radius = local_radius

        # token merging and positional embeddings
        self.token_embed = nn.ModuleList()
        self.encoder_pos_embeds = nn.ModuleList()
        for i in range(len(self.encoder_dims)):
            if i == 0:
                self.token_embed.append(Token_Embed(in_c=3, out_c=self.encoder_dims[i]))
            else:
                self.token_embed.append(Token_Embed(in_c=self.encoder_dims[i - 1], out_c=self.encoder_dims[i]))
            
            self.encoder_pos_embeds.append(nn.Sequential(
                            nn.Linear(3, self.encoder_dims[i]),
                            nn.GELU(),
                            nn.Linear(self.encoder_dims[i], self.encoder_dims[i]),
                        ))

        # encoder blocks
        self.encoder_blocks = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(self.encoder_depths))]
        for i in range(len(self.encoder_depths)):
            self.encoder_blocks.append(Encoder_Block(
                            embed_dim=self.encoder_dims[i],
                            depth=self.encoder_depths[i],
                            drop_path_rate=dpr[depth_count: depth_count + self.encoder_depths[i]],
                            num_heads=self.encoder_num_heads,
                        ))
            depth_count += self.encoder_depths[i]

        self.encoder_norms = nn.ModuleList()
        for i in range(len(self.encoder_depths)):
            self.encoder_norms.append(nn.LayerNorm(self.encoder_dims[i]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def local_att_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(self, neighborhoods, centers, idxs, eval=False):
        # hierarchical encoding
        x_vis_list = []
        xyz_dist = None
        for i in range(len(centers)):
            # 1st-layer encoder, conduct token embedding
            if i == 0:
                group_input_tokens = self.token_embed[i](neighborhoods[0])
            # intermediate layers, conduct token merging
            else:
                b, g1, _ = x_vis.shape
                b, g2, k2, _ = neighborhoods[i].shape
                x_vis_neighborhoods = x_vis.reshape(b * g1, -1)[idxs[i], :].reshape(b, g2, k2, -1)
                group_input_tokens = self.token_embed[i](x_vis_neighborhoods)
            
            if self.local_radius[i] > 0:
                mask_radius, xyz_dist = self.local_att_mask(centers[i], self.local_radius[i], xyz_dist)
                mask_vis_att = mask_radius  #i 0 : [2, 1024, 1024]
            else:
                mask_vis_att = None

            pos = self.encoder_pos_embeds[i](centers[i])
            x_vis = self.encoder_blocks[i](group_input_tokens, pos, mask_vis_att)
            x_vis_list.append(x_vis)

        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.encoder_norms[i](x_vis_list[i]).transpose(-1, -2).contiguous()
        return x_vis_list

class PointNet(nn.Module):
    def __init__(self, input_ch = 3): 
        super(PointNet, self).__init__()
        self.sa1 = PointNetSetAbstraction(256, 0.2, 32, input_ch + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(64, 0.4, 32, 64 + 3, [64, 64, 128], False)
        self.fp2 = PointNetFeaturePropagation(192, [128, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 1, 1)
    def forward(self, xyz):                            
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) 
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)      
        x = self.conv1(l0_points)     
        return x

class PointAttention(nn.Module):
    def __init__(self, channel, reduction=4):   # channel = 1024
        super(PointAttention, self).__init__()
        # 这里可以考虑将 64 -> 1 的卷积换成平均池化再过bn和relu
        self.fcn_1 = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1), 
            nn.BatchNorm1d(channel // reduction), 
            nn.ReLU(),
            nn.Conv1d(channel // reduction, 1, 1), 
            nn.BatchNorm1d(1)
        )
        self.fcn_2 = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1), 
            nn.BatchNorm1d(channel // reduction), 
            nn.ReLU(),
            nn.Conv1d(channel // reduction, 1, 1), 
            nn.BatchNorm1d(1)
        )
        self.fcn_3 = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1), 
            nn.BatchNorm1d(channel // reduction), 
            nn.ReLU(),
            nn.Conv1d(channel // reduction, 1, 1), 
            nn.BatchNorm1d(1)
        )

    def forward(self, feature1, feature2, feature3):
        feature1 = feature1.unsqueeze(dim=1)
        feature2 = feature2.unsqueeze(dim=1)
        feature3 = feature3.unsqueeze(dim=1)
        features = torch.cat((feature1, feature2, feature3), dim=1)  # (B, 3, C, N)
        feature_U = torch.sum(features, dim=1)  # (B, C, N)

        # 计算注意力权重
        a = self.fcn_1(feature_U)   # (B, 1, N)
        b = self.fcn_2(feature_U)   # (B, 1, N)
        c = self.fcn_3(feature_U)   # (B, 1, N)
        matrix = torch.cat((a, b, c), dim=1)   # (B, 3, N)
        matrix = F.softmax(matrix, dim=1)      # (B, 3, N)
        matrix = matrix.unsqueeze(dim=2)       # (B, 3, 1, N)
        features = (matrix * features).sum(dim=1)  # (B, C, N)

        return features


# finetune model
class TransFArchNet(nn.Module):
    def __init__(self,config = None, cls = 3, ):
        super().__init__()
        # self.trans_dim = 384
        # self.group_sizes = [16, 8, 8]
        # self.num_groups = [512, 256, 64]
        # self.encoder_dims = [96, 192, 384]
        
        # self.group_sizes = [32, 16, 16]
        # self.num_groups = [1024, 512, 128]
        # self.encoder_dims = [96, 192, 384]
        
        self.group_sizes = config.group_sizes
        self.num_groups = config.num_groups
        self.encoder_dims = config.encoder_dims
        
        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        # hierarchical encoder
        self.h_encoder = H_Encoder_seg(encoder_dims = self.encoder_dims)

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(0.2))

        self.propagations = nn.ModuleList()
 
        
        self.propagations.append(PointNetFeaturePropagation_(in_channel=self.encoder_dims[0] + self.encoder_dims[1], mlp=[self.encoder_dims[1],self.encoder_dims[1]]))
        self.propagations.append(PointNetFeaturePropagation_(in_channel=self.encoder_dims[2] + self.encoder_dims[1], mlp=[self.encoder_dims[1],self.encoder_dims[1]]))
        

        
        self.PointAttention = PointAttention(channel=self.encoder_dims[1], reduction=4)

        self.offesethead = nn.Sequential(
            nn.Conv1d(self.encoder_dims[1] + 3, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, cls, 1))

    def print_parameters_freeze_status(self):
        for name, param in self.named_parameters():
            print(f"{name}: {'unfrozen' if param.requires_grad else 'frozen'}")
        
    def freeze_backbone(self, freeze_except='weightHead'):
        for name, param in self.named_parameters():
            if freeze_except not in name:
                param.requires_grad = False
        # self.print_parameters_freeze_status()
        
    def unfreeze_all_parameters(self):
        for param in self.parameters():
            param.requires_grad = True

    def load_model_from_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path)
        incompatible = self.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print_log('missing_keys', logger='Point_M2AE_face')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Point_M2AE_face'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Point_M2AE_face')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Point_M2AE_face'
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        # B, C, N = pts.shape
        # pts = pts.transpose(-1, -2).contiguous() # B N 3
        # divide the point cloud in the same form. This is important
        B, N, C = pts.shape
        neighborhoods, centers, idxs = [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)  # b*g*k
    

    
        # 512 point
        # hierarchical encoder
        x_vis_list = self.h_encoder(neighborhoods, centers, idxs, eval=True)
        x_vis_0 =  self.propagations[0](centers[1].transpose(-1, -2), centers[0].transpose(-1, -2), x_vis_list[1], x_vis_list[0])
        x_vis_2 =  self.propagations[1](centers[1].transpose(-1, -2), centers[2].transpose(-1, -2), x_vis_list[1], x_vis_list[2])
        
        fusion_x  = self.PointAttention(x_vis_0,x_vis_list[1],x_vis_2)
        X = torch.cat((fusion_x, centers[1].transpose(-1, -2)), dim=1)  # bs 768 1024
        x = self.offesethead(X)  # bs 3 N
        return centers[1].transpose(-1, -2), x #, d



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss