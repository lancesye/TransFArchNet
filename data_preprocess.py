# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:05:22 2022

@author: lin
"""
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from vedo import *
from scipy.spatial import distance_matrix
import vedo
import random
import pathlib
import re
import vtk
from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

def get_line_data(Y):
    fo1 = open(Y, "r")
    ctrp_list = []
    for l in fo1.readlines():
        lines = l.split()
        if(not len(lines)):
            continue
        str_list = re.findall(r"\d+\.?\d*",lines[0])
        p_list = [float(x) for x in str_list]
        ctrp_list.append(p_list)
    # assert len(ctrp_list) == 11
    # if len(ctrp_list) != 11 :
    #     print(f"error data{Y[id]}")
    ctrp_list = np.array(ctrp_list)
    return ctrp_list

def sample_ctr_to_densePoint(ctr_point,sample_point = 1000):
    control_points = vtk.vtkPoints()
    for i in range(len(ctr_point)):
        control_points.InsertNextPoint(ctr_point[i,0], ctr_point[i,1], ctr_point[i,2])

    spline = vtk.vtkParametricSpline()
    
    xSpline = vtk.vtkKochanekSpline()
    ySpline = vtk.vtkKochanekSpline()
    zSpline = vtk.vtkKochanekSpline()
    
    spline = vtk.vtkParametricSpline()
 
    spline.SetXSpline(xSpline)  #
    spline.SetYSpline(ySpline)
    spline.SetZSpline(zSpline)
    spline.SetPoints(control_points)
    
    function_source = vtk.vtkParametricFunctionSource()
    function_source.SetParametricFunction(spline)
    function_source.SetUResolution(sample_point)
    function_source.Update()
    interpolated_polydata = function_source.GetOutput()
    interpolated_points = interpolated_polydata.GetPoints()
    points_data = interpolated_points.GetData()
    numpy_points = vtk_to_numpy(points_data)
    return numpy_points

# def compute_displacement_map(x1,x2, k = 1):  # x1 npt > x2 npt
#     x1 = x1.permute(0,2,1)
#     x2 = x2.permute(0,2,1)  #[b, np, ch]
#     pairwise_distance = torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
#     idx = pairwise_distance.topk(k=k, dim=2,largest=False)[1]     #[1, 10003, 1]
#     idx = idx[...,0]
#     target_point = index_points(x2, idx)
#     offset = target_point - x1
#     return offset.permute(0,2,1)

def compute_bone_dist(mp,cp, k = 1):  # x1 npt > x2 npt
    x1 = torch.from_numpy(mp).cuda().unsqueeze(0).float()  #[1, 10003, 3]
    x2 = torch.from_numpy(cp).cuda().unsqueeze(0).float()  #[1, 777, 3]
    pairwise_distance = torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
    idx = pairwise_distance.topk(k=k, dim=2,largest=False)[1]     #[1, 10003, 1]
    closest_distances = pairwise_distance.gather(2, idx) 
    return closest_distances.squeeze().cpu().numpy()

def compute_displacement_map(mp,cp, k = 1):  # x1 npt > x2 npt
    x1 = torch.from_numpy(mp).cuda().unsqueeze(0).float()  #[1, 10003, 3]
    x2 = torch.from_numpy(cp).cuda().unsqueeze(0).float()  #[1, 777, 3]
    pairwise_distance = torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
    idx = pairwise_distance.topk(k=k, dim=2,largest=False)[1]     #[1, 10003, 1]
    landmark =  x2[0][idx.squeeze()].cpu().numpy()
    return landmark - mp

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def process_file(file_paths):

    eps = 4
    target_num = 10000 
    # root = 'J:/dataset/dental/CBCT_Mesh/same_direction_bone_npy/'  
    root= 'J:/dataset/dental/CBCT_Mesh/same_direction_down_bone_npy/' 
    ply_paths, bone_paths, ctr_paths, allp_paths = file_paths

    verts, faces = load_ply(str(ply_paths))
    meshes = Meshes(verts=[verts], faces=[faces])
    save_file_name = ply_paths.stem
    print(f"processing:{save_file_name}")

    sampled_points_t, normals_t = sample_points_from_meshes(meshes, target_num, return_normals=True)
    face_sampled_points = sampled_points_t.squeeze().numpy().copy()
    normals = normals_t.squeeze().numpy().copy()
    print(face_sampled_points.shape)

    vedo_mesh  = vedo.load(str(bone_paths))
    verts = vedo_mesh.points()
    faces = vedo_mesh.faces()
    verts = torch.tensor(verts, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int64)
    
    bone_meshes = Meshes(verts=[verts], faces=[faces])
    sampled_points_t, normals_t = sample_points_from_meshes(bone_meshes, target_num*5, return_normals=True)
    bone_sampled_points = sampled_points_t.squeeze().numpy().copy()
    # normals = normals_t.squeeze().numpy().copy()
    print(bone_sampled_points.shape)
    
    bone_distance = compute_bone_dist(face_sampled_points,bone_sampled_points, k = 1)
    
    gt_ctrp = np.loadtxt(str(ctr_paths))
    allp_list = np.loadtxt(str(allp_paths))
    
    mean_cell_centers = np.mean(face_sampled_points, axis=0)
    face_sampled_points[:, 0:3] -= mean_cell_centers[0:3]
    allp_list[:, 0:3] -= mean_cell_centers[0:3]
    gt_ctrp[:, 0:3] -= mean_cell_centers[0:3]
    
    bone_sampled_points[:, 0:3] -= mean_cell_centers[0:3]
     
    maxC = np.max(np.sqrt(np.sum(face_sampled_points ** 2, axis=1)))
    face_sampled_points[:, :3] = face_sampled_points[:, :3] / maxC
    allp_list[:, 0:3] =  allp_list[:, 0:3] / maxC
    gt_ctrp[:, 0:3] = gt_ctrp[:, 0:3]/ maxC
    bone_sampled_points[:, 0:3] = bone_sampled_points[:, 0:3] / maxC
    bone_distance = bone_distance / maxC
    # nmeans_f = normals.mean(axis=0)
    # nstds_f = normals.std(axis=0)
    # for i in range(3):
    #     normals[:,i] = (normals[:,i] - nmeans_f[i]) / nstds_f[i]
        
    # print(barycenters.shape)
    # vedo.show(vedo.Points(barycenters),vedo.Points(allp_list))
    # vedo.close()

    points_feature = np.concatenate((face_sampled_points, normals), axis=1).astype('float32') #15  
    offset = compute_displacement_map(face_sampled_points,allp_list, k = 1)
    
    # face_p_vedo = vedo.Points(points_feature[:,:3]).c('red')#.ps(10)
    # gt_allp_vedo = vedo.Points(allp_list).c('blue')#.ps(10)
    # gt_ctrp_vedo = vedo.Points(gt_ctrp).c('green').ps(10)
    # bone_sampled_points_vedo = vedo.Points(bone_sampled_points).c('black').ps(5)
    # vedo.show(face_p_vedo,bone_sampled_points_vedo, gt_allp_vedo,gt_ctrp_vedo)  
    # vedo.close()
        
    points_feature = points_feature.transpose(1,0) #[1, 15, 16000]
    offset = offset.transpose(1,0)  #[3, 10000]  
    
    sample = {'X': points_feature, 'Y': offset, "ctrp": gt_ctrp, "allp": allp_list, 'bone_points': bone_sampled_points, 'bone_distance': bone_distance, "mean_cell_centers": mean_cell_centers, "max" : maxC}  
    # np.save(root + "align_f1024_npy/" +  save_file_name, sample)
    np.save(os.path.join(root, save_file_name), sample)
    return sample  # 根据实际情况调整返回值s

from concurrent.futures import ProcessPoolExecutor
import pathlib

def process_file_safe(file_path):
    try:
        return process_file(file_path)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def main():
    root = 'J:/dataset/dental/CBCT_Mesh/'
    bone_paths = list(pathlib.Path(root + "same_direction_ply/").glob('*_bone1.vtk'))
    ply_paths = [pathlib.Path(root + "same_direction_ply/" + i.stem[:-6] + '.ply')  for i in bone_paths]
    ctr_curve_list = [pathlib.Path(root + "same_direction_ply/" + i.stem[:-6] + '-ctr.txt')  for i in bone_paths]
    allp_curve_list = [pathlib.Path(root + "same_direction_ply/" + i.stem[:-6] + '-all.txt')  for i in bone_paths]
   
    # bone_paths = list(pathlib.Path(root + "same_direction_ply/").glob('*_bone.vtk'))
    # ply_paths = [pathlib.Path(root + "same_direction_ply/" + i.stem[:-5] + '.ply')  for i in bone_paths]
    # ctr_curve_list = [pathlib.Path(root + "same_direction_ply/" + i.stem[:-5] + '-ctr.txt')  for i in bone_paths]
    # allp_curve_list = [pathlib.Path(root + "same_direction_ply/" + i.stem[:-5] + '-all.txt')  for i in bone_paths]
     
    file_paths = list(zip(ply_paths,bone_paths,ctr_curve_list,allp_curve_list))
    pool_size = 8
    with ProcessPoolExecutor(max_workers=pool_size) as executor:
        # results = list(executor.map(process_file, file_paths))
        results = list(executor.map(process_file_safe, file_paths))
        
if __name__ == "__main__":
    main()