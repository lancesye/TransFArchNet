import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from pathlib import Path
import torch.nn.functional as F
import datetime
import logging
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
import random
import pandas as pd
import pathlib
import re 
from collections import defaultdict
import vedo 
from Adam import Adam

from data_preprocess import index_points
from torch.utils.data import Dataset
import pandas as pd

from vedo import *
from scipy.spatial import distance_matrix
import glob
import random
import pathlib
import vtk
from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
from chamfer_distance import chamfer_distance
import torch.nn as nn
from scipy.interpolate import BSpline, splprep, make_interp_spline,make_lsq_spline
from scipy.spatial import KDTree
from vedo import Points, show

def find_k_nearest_neighbors(points_3d, fpoints_3d, k):
    # 创建KDTree用于高效的最近邻搜索
    tree = KDTree(fpoints_3d)
    
    # 初始化一个列表存储每个牙弓点最近邻的面部点索引
    all_indices = []
    
    # 对于每个牙弓点找到最近邻的K个面部点
    for point in points_3d:
        distances, indices = tree.query(point, k=k)
        all_indices.extend(indices)
    
    # 去除重复的点索引
    unique_indices = list(set(all_indices))
    
    return fpoints_3d[unique_indices],unique_indices

def knn_points(p1, p2, K=1):
    batch_size, num_points_1, dim = p1.shape
    num_points_2 = p2.shape[1]
    p1_expand = p1.unsqueeze(2).expand(batch_size, num_points_1, num_points_2, dim)
    p2_expand = p2.unsqueeze(1).expand(batch_size, num_points_1, num_points_2, dim)
    dists = torch.norm(p1_expand - p2_expand, dim=-1)
    knn_dists, knn_idx = torch.topk(dists, K, largest=False, dim=-1)
    return knn_dists, knn_idx

# Function to compute HD90%
def compute_hd90(gt_pcl, pred_pcl):
    gt_pcl = gt_pcl.unsqueeze(0) if gt_pcl.ndim == 2 else gt_pcl
    pred_pcl = pred_pcl.unsqueeze(0) if pred_pcl.ndim == 2 else pred_pcl
    
    knn_p2g_dists, _ = knn_points(pred_pcl, gt_pcl, K=1)
    knn_g2p_dists, _ = knn_points(gt_pcl, pred_pcl, K=1)
    
    knn_p2g_dists = knn_p2g_dists.squeeze(dim=-1)
    knn_g2p_dists = knn_g2p_dists.squeeze(dim=-1)
    
    hd90 = torch.max(torch.quantile(knn_p2g_dists, 0.90), torch.quantile(knn_g2p_dists, 0.90)).cpu().numpy()
    # hd90 = torch.max(knn_p2g_dists, knn_g2p_dists).cpu().numpy()
    return hd90


def compute_displacement_map(x1,x2, k = 1):  # x1 npt > x2 npt
    # x1 = x1.permute(0,2,1)
    x2 = x2.permute(0,2,1)  #[b, np, ch]
    pairwise_distance = torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
    idx = pairwise_distance.topk(k=k, dim=2,largest=False)[1]     #[1, 10003, 1]
    idx = idx[...,0]
    target_point = index_points(x2, idx)
    offset = target_point - x1
    return offset.permute(0,2,1)

def index_near_weight(x1,x2, weight, k = 1):  # x1 npt > x2 npt
    x1 = x1.permute(0,2,1)
    x2 = x2.permute(0,2,1)  #[b, np, ch]
    pairwise_distance = torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
    idx = pairwise_distance.topk(k=k, dim=2,largest=False)[1]     #[1, 10003, 1]
    idx = idx[...,0]
    idx = idx.squeeze()
    w = weight.squeeze()[idx]
    return w

def arch_fit(pre_allp_list, weight=None, sample_num=512):
    # Sort both points and weights if weight is not None
    if weight is None:
        points_sorted = sorted(pre_allp_list[:, :2], key=lambda x: x[0])
    else:
        points_weighted_sorted = sorted(zip(pre_allp_list[:, :2], weight), key=lambda x: x[0][0])
        points_sorted, weight_sorted = zip(*points_weighted_sorted)
        weight_sorted = np.array(weight_sorted)  # Convert sorted weights back to numpy array
    x = [i[0] for i in points_sorted]
    y = [i[1] for i in points_sorted]
    start_index = int(len(x) * 0.005)
    if start_index < 1:
        start_index = 1
    xs = np.linspace(x[start_index], x[-start_index], sample_num)
    k = 3  # Degree of the spline
    n_t = 11  # Number of knots, t = n + k + 1
    t = np.linspace(x[start_index], x[-start_index], n_t)[1:-1]  # Remove the first and last elements
    t = np.r_[(x[0],) * (k + 1), t, (x[-1],) * (k + 1)]
    # Use weights for fitting if provided
    if weight is None:
        spline = make_lsq_spline(x, y, t, k=k)
    else:
        spline = make_lsq_spline(x, y, t, k=k, w= 1/(weight_sorted**2 + 1e-6))
    ys = spline(xs)
    zs = np.mean(pre_allp_list[:, 2])
    zs = np.broadcast_to(zs, xs.shape)
    fit_pre_allp = np.asarray([xs, ys, zs]).T
    return fit_pre_allp



def extract_roi_points(X, Y, z_min, z_max, distance=None):
    Z_avg = np.mean(Y[:, 2])
    z_min = Z_avg - z_min
    z_max = Z_avg + z_max
  
    
    indices = (X[:, 2] >= z_min) & (X[:, 2] <= z_max)
    
    print(f"original face {X[:, 2].max()} {X[:, 2].min()}")
    print(f"crop face {z_max} {z_min}")
    
    # 提取满足条件的点
    X_roi = X[indices]
    Y_roi = Y[indices]
    
    if distance is None:
        return X_roi, Y_roi
    else:
        distance_roi = distance[indices]
        return X_roi, Y_roi, distance_roi
    


def get_sampling_indices(fpoints_3d, face_points):
    # 创建原始点集的KDTree
    tree = KDTree(fpoints_3d)
    
    # 查找每个采样点的最近邻索引
    _, indices = tree.query(face_points)
    
    return indices

sum_dist_2D = []
sum_dist_3D = []
sum_dist_Z = []
sum_hd90_XY = []
sum_hd90_XYZ = []
case_score_dict = {}


predict_path_txt = r'J:\BaiduSyncdisk\3D-Dental-Segmentation\paper\Face_arch\CBCTMesh2arch\experiment\align_10000P_FeatureFusion_PointM2AESEG_i3_o3_p512_30R_0.2T_0.2S_0.004LR_Adam_CWRST_L1_bs10_ep4090\outputs'
output_path = r'J:\BaiduSyncdisk\3D-Dental-Segmentation\paper\Face_arch\CBCTMesh2arch\experiment\align_10000P_FeatureFusion_PointM2AESEG_i3_o3_p512_30R_0.2T_0.2S_0.004LR_Adam_CWRST_L1_bs10_ep4090'
gt_path = r'J:\BaiduSyncdisk\3D-Dental-Segmentation\paper\Face_arch\CBCTMesh2arch\experiment\align_10000P_train_boneNet_i3_o1_pfull_30R_0.2T_0.2S_0.004LR_Adam_CWRST_L1_bs10_ep4090\outputs'

# 获取 predict_path_txt 下所有 .txt 文件的文件名
predict_txt_files = [f for f in os.listdir(predict_path_txt) if f.endswith('.txt') and not f.endswith('fit.txt') and not f.endswith('_face.txt')]

# predict_face_txt_files = [f for f in os.listdir(predict_path_txt) if f.endswith('.txt') and f.endswith('_face.txt')]

# 遍历所有 .txt 文件并处理对应的 .npy 文件
k = 128
print(len(predict_txt_files))

for txt_file in predict_txt_files:
    txt_file_path = os.path.join(predict_path_txt, txt_file)
    npy_file_name = os.path.splitext(txt_file)[0] + '.npy'
   
    file_name = txt_file
    
    face_point_path = os.path.join(predict_path_txt, txt_file[:-4] + '_face.txt')
    

    
    arch_points_3d = np.loadtxt(txt_file_path)
    fpoints_3d = np.loadtxt(face_point_path)
    # print(arch_points_3d.shape)
    # 加载字典并获取特定键的值
    
    npy_file_path = os.path.join(gt_path, npy_file_name)
    print(npy_file_name)
    if not os.path.exists(npy_file_path):
        print(f"Warning: {npy_file_path} does not exist.")
        continue
    data_dict = np.load(npy_file_path, allow_pickle=True).item()
    face_points = data_dict.get('X')
    ctrp = data_dict.get('ctrp')
    allp = data_dict.get('allp')
    maxp = data_dict.get('max')
    bone_distance = data_dict.get('bone_distance')
    

    # gt_offset = compute_displacement_map(torch.from_numpy(arch_points_3d).unsqueeze(0).float(),allp.float())
    # scalars = torch.norm(gt_offset,dim=1).numpy()[0]
    # print(scalars.max())
    # print(scalars.min())
    # cloud = vtk.vtkPolyData()
    # points_array = numpy_to_vtk(fpoints_3d)
    # scalars_array = numpy_to_vtk(scalars)
    # # 创建vtkPoints对象，并将数据添加到其中
    # vtk_points = vtk.vtkPoints()
    # vtk_points.SetData(points_array)
    # cloud.SetPoints(vtk_points)
    # # 添加标量属性到点云数据
    # cloud.GetPointData().SetScalars(scalars_array)
    # # 保存为VTK格式
    # writer = vtk.vtkPolyDataWriter()
    # writer.SetFileName(predict_path_txt + '/' +  file_name[:-4] + "_error.vtk")
    # writer.SetInputData(cloud)
    # writer.Write()
        
    
    # pred_fit_allp = arch_fit(arch_points_3d, sample_num = 512)
    
    
    # knn_fpoints_3d,face_sample_index = find_k_nearest_neighbors(pred_allp, fpoints_3d, k)
    # s_arch_points_3d  =  arch_points_3d[face_sample_index]
    
    face_sample_index = get_sampling_indices(face_points,fpoints_3d)
    bone_distance = bone_distance[face_sample_index]
    # error
    knn_fpoints_3d, s_arch_points_3d, bone_distance = extract_roi_points(fpoints_3d, arch_points_3d, z_min = maxp.squeeze().numpy()*0.4, z_max = maxp.squeeze().numpy()*0.3,distance = bone_distance)
            

    pred_allp = s_arch_points_3d
    
    print(pred_allp.shape)
    np.savetxt(output_path + '/' +  file_name[0]+ "_filter.txt", pred_allp)
    points = vedo.Points(pred_allp)
    vedo.write(points, output_path + '/' +  file_name[0]+ "_filter.ply")
    
    pred_allp = arch_fit(pred_allp,weight = bone_distance, sample_num = 512)
    np.savetxt(output_path + '/' +  file_name[0]+ "_filter_fit.txt", pred_allp)
    points = vedo.Points(pred_allp)
    vedo.write(points, output_path + '/' +  file_name[0]+ "_filter_fit.ply")
    
    
    pred_3d = torch.from_numpy(pred_allp).unsqueeze(0).float().cuda()
    allp_3d = allp.permute(0,2,1).float().cuda()
    dist_3d, _ = chamfer_distance(pred_3d, allp_3d)
    dist_2d, _ = chamfer_distance(pred_3d[:,:,:2], allp_3d[:,:,:2])
    z_dist = torch.abs(pred_3d[:,:,2]- allp_3d[:,0,2].unsqueeze(0)).mean()
    sum_dist_3D.append(dist_3d.item())
    sum_dist_2D.append(dist_2d.item())
    sum_dist_Z.append(z_dist.item())

    hd90_XY = compute_hd90(allp_3d[:, :, :2], pred_3d[:, :, :2])
    hd90_XYZ = compute_hd90(allp_3d, pred_3d)
    sum_hd90_XY.append(hd90_XY.item())
    sum_hd90_XYZ.append(hd90_XYZ.item())

    #TODO 保留分数字典 排序 
    case_score_dict.update({f"{file_name}":[dist_3d.item(),dist_2d.item(),z_dist.item(), hd90_XY.item(), hd90_XYZ.item()]})
        
sorted_dict = dict(sorted(case_score_dict.items(), key=lambda item: sum(item[1]), reverse=True))
sorted_dict.update({"3D bidirectional distance error mean std max min": [np.mean(sum_dist_3D), np.std(sum_dist_3D), np.max(sum_dist_3D), np.min(sum_dist_3D)]})
sorted_dict.update({"2D bidirectional distance error mean std max min": [np.mean(sum_dist_2D), np.std(sum_dist_2D), np.max(sum_dist_2D), np.min(sum_dist_2D)]})
sorted_dict.update({"Z distance error mean std max min": [np.mean(sum_dist_Z), np.std(sum_dist_Z), np.max(sum_dist_Z), np.min(sum_dist_Z)]})
sorted_dict.update({"HD90 XY mean std max min": [np.mean(sum_hd90_XY), np.std(sum_hd90_XY), np.max(sum_hd90_XY), np.min(sum_hd90_XY)]})
sorted_dict.update({"HD90 XYZ mean std max min": [np.mean(sum_hd90_XYZ), np.std(sum_hd90_XYZ), np.max(sum_hd90_XYZ), np.min(sum_hd90_XYZ)]})

sorted_dict.update({"experiment":str(predict_path_txt)})
sorted_dict.update({"test dataset number": [len(predict_txt_files)]})

print(sorted_dict)
import json
json_str = json.dumps(sorted_dict, indent=4)
score_file = str(predict_path_txt)
with open(os.path.join(score_file, f'bonew_fit_sorted_scores.json'), 'w') as f:
    f.write(json_str)

