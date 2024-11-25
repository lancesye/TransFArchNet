
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
import vedo
from model.curvenet_util import index_points
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from vedo import *
from scipy.spatial import distance_matrix
import glob
import random
import pathlib
import vtk
from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
from chamfer_distance import chamfer_distance
import torch.nn as nn
import hydra, logging
from omegaconf import OmegaConf 
from TransFArchNet import TransFArchNet
from scipy.interpolate import BSpline, splprep, make_interp_spline,make_lsq_spline
from scipy.spatial import KDTree

# def extract_roi_points(X, Y, z_min, z_max):
#     Z_avg = torch.mean(Y[:, 2, :], dim=1, keepdim=True)
#     z_min = Z_avg - z_min  # [batch_size, 1, 1]
#     z_max = Z_avg + z_max  # [batch_size, 1, 1]
#     indices = (X[0, 2, :] >= z_min) & (X[0, 2, :] <= z_max)
#             # 提取满足条件的点
#     X_roi = X[0, :, indices[0]].unsqueeze(0)
#     Y_roi = Y[0, :, indices[0]].unsqueeze(0)
#     return X_roi, Y_roi

def get_sampling_indices(fpoints_3d, face_points):
    # 创建原始点集的KDTree
    tree = KDTree(fpoints_3d)
    
    # 查找每个采样点的最近邻索引
    _, indices = tree.query(face_points)
    
    return indices


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
    xs = np.linspace(x[start_index], x[-start_index], sample_num)
    k = 3  # Degree of the spline
    n_t = 11  # Number of knots, t = n + k + 1
    t = np.linspace(x[start_index], x[-start_index], n_t)[1:-1]  # Remove the first and last elements
    t = np.r_[(x[0],) * (k + 1), t, (x[-1],) * (k + 1)]
    # Use weights for fitting if provided
    if weight is None:
        spline = make_lsq_spline(x, y, t, k=k)
    else:
        spline = make_lsq_spline(x, y, t, k=k, w= 1/weight_sorted)
    ys = spline(xs)
    zs = np.mean(pre_allp_list[:, 2])
    zs = np.broadcast_to(zs, xs.shape)
    fit_pre_allp = np.asarray([xs, ys, zs]).T
    return fit_pre_allp

def extract_roi_points(X, Y, z_min, z_max, distance=None):
    Z_avg = np.mean(Y[:, 2])
    z_min = Z_avg - z_min
    z_max = Z_avg + z_max
    # z_min = Z_avg - 30
    # z_max = Z_avg + 40
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
    


def set_rand_seed(seed=42):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # # torch.backends.cudnn.enabled = False       
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True   # 保证每次返回得的卷积算法是确定的

def get_order(file):
    file_pattern = re.compile(r'.*?(\d+).*?')
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])



def load_checkpoint(checkpoint_path, model, optimizer):
    """
    加载检查点并恢复训练状态。
    """
    # 检查检查点文件是否存在
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['losses']
        print(f"成功加载检查点 '{checkpoint_path}'，继续训练从epoch {epoch}。")
    except FileNotFoundError:
        # 如果文件不存在，从头开始训练
        epoch = 0
        loss = np.inf
        print(f"没有找到检查点 '{checkpoint_path}'，从头开始训练。")
    return epoch, loss

def load_mesh_and_preprocess(ply_path, target_num=10000, eps=4):
    mesh = vedo.load(str(ply_path))
    
    # Decimate mesh to a specific number of points
    ratio = (target_num + eps) / mesh.ncells  # Calculate the decimation ratio
    mesh.decimate(fraction=ratio)
    
    assert mesh.ncells > target_num, "The mesh does not have enough cells after decimation."
    
    # Extract vertices and cell centers (barycenters)
    vertices = mesh.points()
    barycenters = mesh.cell_centers.copy()
    
    # Compute center of mass and normalize vertices
    mean_cell_centers = mesh.center_of_mass()
    vertices -= mean_cell_centers  # Subtract center of mass from all points
    barycenters -= mean_cell_centers
    
    # Scale normalization
    max_dist = np.max(np.sqrt(np.sum(vertices ** 2, axis=1)))
    vertices /= max_dist
    barycenters /= max_dist
    
    return barycenters[:target_num], mean_cell_centers, max_dist,  ply_path.stem



@hydra.main(config_path="face.yaml")  #
def train_app(cfg):
    set_rand_seed()
    # """-------------------------- parameters --------------------------------------"""
    batch_size = 10
    pretrain = False
    data_root = 'I:/align_s512_npy/'
    fp16 = False
    input_ch = 3
    out_ch = 3
    lr = 4e-3
    epoch = 4000
    point = 512
    # """--------------------------- create Folder ----------------------------------"""
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + f'/align_10000P_FeatureFusion_TransFArchNet_i{input_ch}_o{out_ch}_p{point}_30R_0.2T_0.2S_{lr}LR_Adam_CWRST_L1_bs{batch_size}_ep{epoch}')  #_STN15d
    file_dir.mkdir(exist_ok=True)
    log_dir, checkpoints = file_dir.joinpath('logs/'), file_dir.joinpath('checkpoints')
    log_dir.mkdir(exist_ok=True)
    checkpoints.mkdir(exist_ok=True)
    checkpoint_name = 'latest_checkpoint.tar'
    OmegaConf.save(cfg, str(file_dir) + '/face.yaml')
    #%%
    formatter = logging.Formatter('%(name)s - %(message)s')
    logger = logging.getLogger("all")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(str(log_dir) + '/log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    writer = SummaryWriter(file_dir.joinpath('tensorboard'))
    output_path = str(file_dir)


    model = TransFArchNet(cfg.model).cuda()  #
    bone_model = PointNetPlus_Seg(3,1).cuda()
    
    start_epoch = 0
    # if(pretrain):
    #     checkpoint_path =  "experiment/align_10000P_PointM2AESEG_i3_o3_p512_30R_0.2T_0.2S_0.004LR_Adam_CWRST_CDL2_bs10_ep4096/checkpoints/latest_checkpoint.tar"
    #     # model_dict = model.state_dict()
    #     # pretrained_dict  = torch.load(checkpoint_path, map_location='cpu')
    #     # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     # model_dict.update(pretrained_dict)
    #     # model.load_state_dict(model_dict) #checkpoint['model_state_dict']
    #     start_epoch, loss = load_checkpoint(checkpoint_path, model, optimizer)
    #     logger.info("load pretrained_dict from" + checkpoint_path)
    #     print("load pretrained_dict from" + checkpoint_path)

    model.cuda()
    # """------------------------------------- test --------------------------------"""
    output_path = str(file_dir) + '/outputs'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    model_list = list(pathlib.Path(checkpoints).glob('*.pth'))
    model_list = sorted(model_list, key=get_order)
    model_path = model_list[-1]

    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load trained model
    checkpoint = torch.load(os.path.join(model_path), map_location='cpu') #join(checkpoints, model_name)
    # model.load_state_dict(checkpoint) #checkpoint['model_state_dict']

    model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})


    model = model.to(device, dtype=torch.float)
    
    bone_chechpoint = torch.load('experiment/align_10000P_train_boneNet_i3_o1_pful/checkpoints/coordinate_812_0.019050.pth')
    bone_model.load_state_dict(bone_chechpoint) 
    print("load best model")
    print(model_path)


    model.eval()
    bone_model.eval()
    sum_dist_2D = []
    sum_dist_3D = []
    sum_dist_Z = []
    sum_hd90_XY = []
    sum_hd90_XYZ = []
    case_score_dict = {}
    
    
    mesh_folder = r'J:\BaiduSyncdisk\3D-Dental-Segmentation\paper\Face_arch\Transformed_Meshes'

    mesh_files = list(Path(mesh_folder).glob('*.obj'))
    mesh_files.sort()
    output_path = mesh_folder
    print(f'Found {len(mesh_files)} mesh files.')


    for ply_file in mesh_files:
        barycenters, mean_cell_centers, maxp,file_name = load_mesh_and_preprocess(ply_file)
        
                
        X = torch.tensor(barycenters, dtype=torch.float32).T.cuda()

        X = X.unsqueeze(0)
        X_copy = X.clone()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled = fp16):
                    xyz, pre_offset  = model(X.permute(0,2,1).contiguous())                 #idx [b,1024]
                    pred = xyz + pre_offset
                    bone_distance = bone_model(X_copy)   
        
        bone_distance = bone_distance.squeeze().detach().cpu().numpy() * maxp
        pred_ctrp= (pred.detach().cpu().numpy()*maxp)[0] + np.expand_dims(mean_cell_centers,axis=1)
        pred_curve_np = pred_ctrp#.detach().numpy() 

        pred_allp = pred_curve_np.transpose(1,0)
        
        print(pred_allp.shape)
        # face_points= (xyz.cpu()*maxp + mean_cell_centers).numpy()[0].transpose(1,0)
        face_points= ((xyz.detach().cpu().numpy()*maxp)[0] + np.expand_dims(mean_cell_centers,axis=1)).transpose(1,0)
        # np.savetxt(output_path + '/' +  file_name+ "_face.txt", face_points)
        # points = vedo.Points(face_points)
        # vedo.write(points, output_path + '/' +  file_name+ "_face.ply")
        
        np.savetxt(output_path + '/' +  file_name+ ".txt", pred_allp)
        points = vedo.Points(pred_allp)
        vedo.write(points, output_path + '/' +  file_name+ ".ply")
        
        pred_allp_fit = arch_fit(pred_allp,sample_num = 512)
        np.savetxt(output_path + '/' +  file_name+ "_fit.txt", pred_allp_fit)
        points = vedo.Points(pred_allp_fit)
        vedo.write(points, output_path + '/' +  file_name+ "_fit.ply")
        
        fpoints_3d = xyz.squeeze().detach().cpu().numpy()*maxp + np.expand_dims(mean_cell_centers,axis=1)
        face_sample_index = get_sampling_indices(face_points,fpoints_3d.transpose(1,0))
        bone_distance = bone_distance[face_sample_index]
        # error
        knn_fpoints_3d, s_arch_points_3d, bone_distance = extract_roi_points(fpoints_3d.transpose(1,0), pred_allp, z_min = maxp*0.4, z_max = maxp*0.2,distance = bone_distance)
                
        pred_allp = s_arch_points_3d
    
        print(pred_allp.shape)
        np.savetxt(output_path + '/' +  file_name+ "_crop.txt", pred_allp)
        points = vedo.Points(pred_allp)
        vedo.write(points, output_path + '/' +  file_name+ "_crop.ply")
        
        pred_allp = arch_fit(pred_allp,weight = bone_distance, sample_num = 512)
        np.savetxt(output_path + '/' +  file_name+ "_crop_fit.txt", pred_allp)
        points = vedo.Points(pred_allp)
        vedo.write(points, output_path + '/' +  file_name+ "_crop_fit.ply")
        
        
    

if __name__ == "__main__":
    train_app()