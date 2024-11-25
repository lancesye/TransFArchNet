
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

from TransFArchNet import TransFArchNet
from chamfer_distance import chamfer_distance
import torch.nn as nn
import hydra, logging
from omegaconf import OmegaConf 
from scipy.interpolate import BSpline, splprep, make_interp_spline,make_lsq_spline

def knn_points(p1, p2, K=1):
    batch_size, num_points_1, dim = p1.shape
    num_points_2 = p2.shape[1]
    p1_expand = p1.unsqueeze(2).expand(batch_size, num_points_1, num_points_2, dim)
    p2_expand = p2.unsqueeze(1).expand(batch_size, num_points_1, num_points_2, dim)
    dists = torch.norm(p1_expand - p2_expand, dim=-1)
    knn_dists, knn_idx = torch.topk(dists, K, largest=False, dim=-1)
    return knn_dists, knn_idx

def extract_roi_points(X, Y, z_min, z_max):
    Z_avg = torch.mean(Y[:, 2, :], dim=1, keepdim=True)
    z_min = Z_avg - z_min  # [batch_size, 1, 1]
    z_max = Z_avg + z_max  # [batch_size, 1, 1]
    indices = (X[0, 2, :] >= z_min) & (X[0, 2, :] <= z_max)
            # 提取满足条件的点
    X_roi = X[0, :, indices[0]].unsqueeze(0)
    Y_roi = Y[0, :, indices[0]].unsqueeze(0)
    return X_roi, Y_roi

# Function to compute HD90%
def compute_hd90(gt_pcl, pred_pcl):
    gt_pcl = gt_pcl.unsqueeze(0) if gt_pcl.ndim == 2 else gt_pcl
    pred_pcl = pred_pcl.unsqueeze(0) if pred_pcl.ndim == 2 else pred_pcl
    
    knn_p2g_dists, _ = knn_points(pred_pcl, gt_pcl, K=1)
    knn_g2p_dists, _ = knn_points(gt_pcl, pred_pcl, K=1)
    
    knn_p2g_dists = knn_p2g_dists.squeeze(dim=-1)
    knn_g2p_dists = knn_g2p_dists.squeeze(dim=-1)
    
    hd90 = torch.max(torch.quantile(knn_p2g_dists, 0.9), torch.quantile(knn_g2p_dists, 0.9)).cpu().numpy()
    return hd90

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

def sample_ctr_to_densePoint(ctr_point,sample_point = 1000):
    # shape N 3
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
        
def weight_init(m):  #初始化权重
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight.data)
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.02)
        # m.bias.data.zero_()

def vtk_trans_point(point_set,Trans):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(point_set))
    pdata  = vtk.vtkPolyData()
    pdata.SetPoints(points)
    Tf = vtk.vtkTransformFilter()
    Tf.SetInputData(pdata)
    Tf.SetTransform(Trans)
    Tf.Update()
    tf_point = Tf.GetOutput().GetPoints()
    tf_ctrp = vtk_to_numpy(tf_point.GetData())
    return tf_ctrp
    
def GetVTKTransformationMatrix(rotate_X=[-30, 30], rotate_Y=[-30, 30], rotate_Z=[-30, 30],
                               translate_X=[-0.2, 0.2], translate_Y=[-0.2, 0.2], translate_Z=[-0.2, 0.2],
                               scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]):
    '''
    get transformation matrix (4*4)
    return: vtkMatrix4x4
    '''
    Trans = vtk.vtkTransform()
    ry_flag = np.random.randint(0,2) #if 0, no rotate
    rx_flag = np.random.randint(0,2) #if 0, no rotate
    rz_flag = np.random.randint(0,2) #if 0, no rotate
    if ry_flag == 1:
        # rotate along Yth axis
        Trans.RotateY(np.random.uniform(rotate_Y[0], rotate_Y[1]))
    if rx_flag == 1:
        # rotate along Xth axis
        Trans.RotateX(np.random.uniform(rotate_X[0], rotate_X[1]))
    if rz_flag == 1:
        # rotate along Zth axis
        Trans.RotateZ(np.random.uniform(rotate_Z[0], rotate_Z[1]))

    trans_flag = np.random.randint(0,2) #if 0, no translate
    if trans_flag == 1:
        Trans.Translate([np.random.uniform(translate_X[0], translate_X[1]),
                            np.random.uniform(translate_Y[0], translate_Y[1]),
                            np.random.uniform(translate_Z[0], translate_Z[1])])
    scale_flag = np.random.randint(0,2)
    if scale_flag == 1:
        Trans.Scale([np.random.uniform(scale_X[0], scale_X[1]),
                     np.random.uniform(scale_Y[0], scale_Y[1]),
                     np.random.uniform(scale_Z[0], scale_Z[1])])
    matrix = Trans.GetMatrix()
    return matrix,Trans

class Mesh_Dataset(Dataset):
    def __init__(self, data_list, using_aug = True):
        self.data_list = data_list
        self.using_aug = using_aug
    def __len__(self):
        return len( self.data_list)

    def __getitem__(self, idx):
        file_path = pathlib.Path(self.data_list[idx])
        smaple = np.load(str(file_path), allow_pickle=True)  
        X = smaple.item()['X'] 
        offset = smaple.item()['Y'] 
        max = smaple.item()['max'] 
        ctrp = smaple.item()['ctrp'] 
        allp = smaple.item()['allp'] 
        # bone_distance = smaple.item()['bone_distance']
        bone_distance = 0
        mean_cell_centers = smaple.item()['mean_cell_centers']  
        # data augmentation
        if(self.using_aug):
            tk_matrix,Trans = GetVTKTransformationMatrix(rotate_X=[-30, 30], rotate_Y=[-30, 30], rotate_Z=[-30, 30],   
                                                        translate_X=[-0.2, 0.2], translate_Y=[-0.2, 0.2], translate_Z=[-0.2, 0.2],
                                                        scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2])
            ctrp = vtk_trans_point(ctrp,Trans)
            allp = vtk_trans_point(allp,Trans)
            X[:3,:] = vtk_trans_point(X[:3,:].transpose(1,0),Trans).transpose(1,0)      
        return X,offset,ctrp.transpose(1,0),allp.transpose(1,0),mean_cell_centers.reshape((3, 1)),max.reshape((1, 1)),file_path.stem #bone_distance,
    
def get_dataset(root = 'J:/dataset/dental/CBCT_Mesh/align_s512_npy/',bs  =2, seed = 42 ,nw = 1):
    random.seed(seed)
    np.random.seed(seed)
    
    # test_list = [os.path.join(root, i + '.npy') for i in test_name]
    img_list = glob.glob(root  + '*.npy')
    img_list.sort()
    num = len(img_list)
    ID = np.arange(num)
    img_list = np.array(img_list)
    np.random.shuffle(ID)
    print('data ID', ID)
    train_split = int(0.6*num)
    val_split = int(0.1*num)
    test_split = int(0.3*num)
    train_ID = ID[:train_split]
    val_ID = ID[train_split:(val_split + train_split)]
    test_ID = ID[(val_split + train_split):]
    train_x = img_list[train_ID]
    val_x = img_list[val_ID]
    test_x = img_list[test_ID]
    print('train num',len(train_x))
    print('val_x num',len(val_x))
    print('test_x num',len(test_x))
    train_dataset = Mesh_Dataset(train_x)
    val_dataset = Mesh_Dataset(val_x,using_aug = False)  
    test_dataset = Mesh_Dataset(test_x,using_aug = False)  
    # test_dataset = Mesh_Dataset(test_list,using_aug = False)  
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=bs,shuffle=True,num_workers = nw)  #, collate_fn=collate_fn  ,
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=bs,shuffle=False,num_workers = nw) #, collate_fn=collate_fn  #,num_workers = 2
    return train_loader,val_loader,test_dataset

def compute_displacement_map(x1,x2, k = 1):  # x1 npt > x2 npt
    x1 = x1.permute(0,2,1)
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

@hydra.main(config_path="cfgs/fine-tuning/face.yaml")  #
def train_app(cfg):
    set_rand_seed()
    # """-------------------------- parameters --------------------------------------"""
    batch_size = 10
    pretrain = False
    # data_root = r'J:\dataset\dental\CBCT_Mesh\align_s512_npy'
    # data_root = '/public/linguoye/dataset/CBCT_Mesh_Curve/align_s512_npy/'
    # data_root = 'J:/dataset/dental/CBCT_Mesh/same_direction_bone_npy/'
    # data_root = 'J:/dataset/dental/CBCT_Mesh/same_direction_down_bone_npy'
    data_root = 'I:/align_s512_npy/'
    fp16 = False
    input_ch = 3
    out_ch = 3
    lr = 4e-3
    epoch = 4090
    point = 1024
    # """--------------------------- create Folder ----------------------------------"""
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + f'/align_10000P_FeatureFusion_TransArchNet_i{input_ch}_o{out_ch}_p{point}_30R_0.2T_0.2S_{lr}LR_Adam_CWRST_L1_bs{batch_size}_ep{epoch}')  #_STN15d
    # file_dir = Path(str(experiment_dir) + f'/align_10000P_PointM2AESEG_i{input_ch}_o{out_ch}_p{point}_30R_0.2T_0.2S_{lr}LR_Adam_CWRST_L1_show_bs{batch_size}_ep{epoch}')  #_STN15d
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
    """-------------------------------- Dataloader --------------------------------"""
    train_loader,val_loader,test_dataset = get_dataset(root = data_root,bs = batch_size)   
    """--------------------------- Build Network and optimizer----------------------"""
    model = TransFArchNet(cfg.model).cuda()  #
    model.apply(weight_init) 

    optimizer = Adam(
        model.parameters(),
        lr=lr,
    )
    # optimizer = torch.optim.Adam(
    # model.parameters(),
    #     lr=1e-2,
    #     betas=(0.9, 0.999),
    #     # weight_decay=1e-5
    # )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=4, T_mult=2, eta_min=5e-5)

    start_epoch = 0
    if(pretrain):
        checkpoint_path =  "experiment/align_10000P_PointM2AESEG_i3_o3_p512_30R_0.2T_0.2S_0.004LR_Adam_CWRST_CDL2_bs10_ep4096/checkpoints/latest_checkpoint.tar"
        # model_dict = model.state_dict()
        # pretrained_dict  = torch.load(checkpoint_path, map_location='cpu')
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict) #checkpoint['model_state_dict']
        start_epoch, loss = load_checkpoint(checkpoint_path, model, optimizer)
        logger.info("load pretrained_dict from" + checkpoint_path)
        print("load pretrained_dict from" + checkpoint_path)

    model.cuda()
    # """------------------------------------- train --------------------------------"""
    logger.info("------------------train------------------")
    best_acc = 1000
    LEARNING_RATE_CLIP = 1e-6
    his_loss = []
    his_smotth = []
    class_weights = torch.ones(15).cuda()
    
    scaler = torch.cuda.amp.GradScaler(enabled= fp16)

    loss_funtion = L1Loss()  #SmoothL1Loss  reduction='none'
    # loss_funtion = L1Loss(reduction='none')  #SmoothL1Loss  reduction='none'
    val_loss_funtion = L1Loss()

    for epoch in range(start_epoch, epoch):
        model.train()
        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        optimizer.param_groups[0]['lr'] = lr

        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            X,offset,ctrp,allp,mean_cell_centers,maxp,file_name = data         #[1, 16000, 24]
            X, offset,allp  = X.float().cuda(),offset.float().cuda(),allp.float().cuda()
            X = X[:,:input_ch]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled = fp16):
                xyz,pre_offset  = model(X.permute(0,2,1).contiguous())                
    #             xyz,pre_offset  = model(X)                 #idx [b,1024]
                # gt_offset = index_points(offset.permute(0,2,1), fps_idx).permute(0,2,1)
                gt_offset = compute_displacement_map(xyz,allp)
                loss = loss_funtion(pre_offset,gt_offset) 
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            his_loss.append(loss.cpu().data.numpy())

        scheduler.step()
        
        
        model.eval()
        val_loss_all = []
        writer.add_scalar("train_loss", np.mean(his_loss), epoch)
        for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
            X,offset,ctrp,allp,mean_cell_centers,maxp,file_name = data         #[1, 16000, 24]
            X, offset,allp = X.float().cuda(),offset.float().cuda(),allp.float().cuda()
            X = X[:,:input_ch]
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled = fp16):
                    xyz,pre_offset   = model(X.permute(0,2,1).contiguous())                 #idx [b,1024]
                    gt_offset = compute_displacement_map(xyz,allp)
                    val_loss = loss_funtion(pre_offset,gt_offset) 

            val_loss_all.append(val_loss.item())
        val_loss = np.mean(val_loss_all)
        print("Epoch: %d, LR: %f, train loss: %f, val loss= %f " % (epoch,lr,np.mean(his_loss), val_loss))

        logger.info("Epoch: %d, LR: %f, train loss: %f, val loss= %f" % (epoch, lr,np.mean(his_loss), val_loss))
        writer.add_scalar("val_loss", val_loss, epoch)


        # save the checkpoint   
        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': np.mean(his_loss)},
                    '%s/%s'%(str(checkpoints),checkpoint_name)
                    )

        if (val_loss < best_acc):
            best_acc = val_loss
            print("lowest loss: %f" % (best_acc))
            logger.info("lowest loss: %f" % (best_acc))
            torch.save(model.state_dict(), '%s/coordinate_%d_%f.pth' % (checkpoints, epoch, best_acc))
            best_pth = '%s/coordinate_%d_%f.pth' % (checkpoints, epoch, best_acc)
        his_loss.clear()
        writer.close()
            
        if epoch % 10 == 0:
            pred = xyz + pre_offset
            pred_ctrp= (pred.cpu()*maxp + mean_cell_centers)
            allp = (allp.cpu() * maxp + mean_cell_centers)[0]
            pred_curve_np = pred_ctrp.detach().numpy()[0]

            allp =  allp.numpy()
            allp = allp.transpose(1,0)
            pred_allp = pred_curve_np.transpose(1,0)
            points = vedo.Points(pred_allp)
            vedo.write(points, output_path + '/' +  str(epoch) + file_name[0]+ "_pred.ply")
            gt_points = vedo.Points(allp)
            vedo.write(gt_points, output_path + '/' +  str(epoch) +   file_name[0]+ "_gt.ply")
                  
 # %%
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
    print("load best model")
    print(model_path)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    model.eval()
    sum_dist_2D = []
    sum_dist_3D = []
    sum_dist_Z = []
    sum_hd90_XY = []
    sum_hd90_XYZ = []
    case_score_dict = {}
    
    for i, data in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
        X,offset,ctrp,allp,mean_cell_centers,maxp,file_name = data         #[1, 16000, 24]
        X = X.float().cuda()
        X = X[:,:input_ch]
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled = fp16):
                    xyz, pre_offset  = model(X.permute(0,2,1).contiguous())                 #idx [b,1024]
                    pred = xyz + pre_offset
                    
        pred_ctrp= (pred.cpu()*maxp + mean_cell_centers)
        ctrp = (ctrp * maxp + mean_cell_centers)
        allp = (allp * maxp + mean_cell_centers)
        
        pred_curve_np = pred_ctrp.detach().numpy()[0]
        
        # pred_allp = sample_ctr_to_densePoint(pred_curve_np.transpose(1,0),sample_point = 511)
        pred_allp = pred_curve_np.transpose(1,0)
        
        print(pred_allp.shape)
        face_points= (xyz.cpu()*maxp + mean_cell_centers).numpy()[0].transpose(1,0)
        np.savetxt(output_path + '/' +  file_name[0]+ "_face.txt", face_points)
        points = vedo.Points(face_points)
        vedo.write(points, output_path + '/' +  file_name[0]+ "_face.ply")
        
        np.savetxt(output_path + '/' +  file_name[0]+ ".txt", pred_allp)
        points = vedo.Points(pred_allp)
        vedo.write(points, output_path + '/' +  file_name[0]+ ".ply")
        
        pred_allp = arch_fit(pred_allp,sample_num = 512)
        np.savetxt(output_path + '/' +  file_name[0]+ "_fit.txt", pred_allp)
        points = vedo.Points(pred_allp)
        vedo.write(points, output_path + '/' +  file_name[0]+ "_fit.ply")
        
        
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

    sorted_dict.update({"experiment":str(file_dir)})
    sorted_dict.update({"test dataset number": [len(test_dataset)]})

    print(sorted_dict)
    import json
    json_str = json.dumps(sorted_dict, indent=4)
    score_file = str(file_dir)
    with open(os.path.join(score_file, 'fit_sorted_scores.json'), 'w') as f:
        f.write(json_str)

    with open(os.path.join(score_file, 'fit_logs.txt'), 'w') as f:
        f.write("3D bidirectional distance error\n")
        f.write(f"Mean: {np.mean(sum_dist_3D)}\n")
        f.write(f"Standard Deviation: {np.std(sum_dist_3D)}\n")
        f.write(f"Max: {np.max(sum_dist_3D)}\n")
        f.write(f"Min: {np.min(sum_dist_3D)}\n")
        f.write("2D bidirectional distance error\n")
        f.write(f"Mean: {np.mean(sum_dist_2D)}\n")
        f.write(f"Standard Deviation: {np.std(sum_dist_2D)}\n")
        f.write(f"Max: {np.max(sum_dist_2D)}\n")
        f.write(f"Min: {np.min(sum_dist_2D)}\n")
        f.write("Z distance error\n")
        f.write(f"Mean: {np.mean(sum_dist_Z)}\n")
        f.write(f"Standard Deviation: {np.std(sum_dist_Z)}\n")
        f.write(f"Max: {np.max(sum_dist_Z)}\n")
        f.write(f"Min: {np.min(sum_dist_Z)}\n")
        f.write("HD90 XY\n")
        f.write(f"Mean: {np.mean(sum_hd90_XY)}\n")
        f.write(f"Standard Deviation: {np.std(sum_hd90_XY)}\n")
        f.write(f"Max: {np.max(sum_hd90_XY)}\n")
        f.write(f"Min: {np.min(sum_hd90_XY)}\n")
        f.write("HD90 XYZ\n")
        f.write(f"Mean: {np.mean(sum_hd90_XYZ)}\n")
        f.write(f"Standard Deviation: {np.std(sum_hd90_XYZ)}\n")
        f.write(f"Max: {np.max(sum_hd90_XYZ)}\n")
        f.write(f"Min: {np.min(sum_hd90_XYZ)}\n")
        
if __name__ == "__main__":
    train_app()