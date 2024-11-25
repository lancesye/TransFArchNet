import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
# from model.
from model.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation,index_points

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class PointNetReg(nn.Module):
    def __init__(self, input_ch =3, output_ch = 33):  # num_point: curve contrl point number 
        super(PointNetReg, self).__init__()

        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, input_ch + 3, [32, 32, 64], False, False)
        # self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, input_ch, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.classfier = nn.Sequential(
            nn.Linear(512*16, 512),
            nn.BatchNorm1d(512),
            Mish(),
            # nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            Mish(),
            nn.Linear(256, output_ch))

    def forward(self, xyz):                              #[2, 15, 5000]
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        l1_xyz, l1_points  = self.sa1(l0_xyz, l0_points)  #o [2, 3, 1024]) [2, 64, 1024]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  #o [2, 3, 512]) [2, 64, 152]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  #o [2, 3, 64]) [2, 64, 64]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  #o [2, 3, 16]) [2, 512, 16]) 
        # l4_points = l4_points.max(dim=2, keepdim=False)[0]    
        l4_points = l4_points.reshape(l4_xyz.shape[0],-1)
        x = self.classfier(l4_points)            
        x = x.reshape(l4_xyz.shape[0],3,-1)
        return x


class PointNetPlus_Seg(nn.Module):
    def __init__(self, input_ch = 16, num_classes=3):
        super(PointNetPlus_Seg, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, input_ch + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):                              #[2, 15, 5000]
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  #o [2, 3, 1024]) [2, 64, 1024]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  #o [2, 3, 512]) [2, 64, 512]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  #o [2, 3, 64]) [2, 64, 64]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  #o [2, 3, 16]) [2, 512, 16]) 

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # o [2, 256, 64]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # o [2, 256, 256]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # o [2, 128, 1024]
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)      # 0 [2, 128, 5000]

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))  # o [2,128,5000]
        x = self.conv2(x)              # o [2,24,5000]
        # x = F.log_softmax(x, dim=1)     
        # x = x.permute(0, 2, 1)         # o [2,5000,24]
        return x

class VoteNet(nn.Module):
    def __init__(self, input_ch = 16, num_classes=3):
        super(VoteNet, self).__init__()
        self.sa1 = PointNetSetAbstraction(2048, 0.2, 64, input_ch + 3, [64, 64, 128], False, True)
        self.sa2 = PointNetSetAbstraction(1024, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(512, 0.8, 16, 256 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(256, 1.2, 16, 256 + 3, [128, 128, 256], False)

        self.fp4 = PointNetFeaturePropagation(512, [256, 256])
        self.fp3 = PointNetFeaturePropagation(512, [256, 256])
        self.fp2 = PointNetFeaturePropagation(384, [256, 256])
        
        # '''feature-wise attention'''
        # self.fa = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
        #                         nn.BatchNorm1d(256),
        #                         nn.LeakyReLU(0.2))

        self.vote = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            Mish(),
            # nn.Dropout(0.5),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            Mish(),
            # nn.Dropout(0.5),
            nn.Conv1d(256, num_classes, 1))

    def forward(self, xyz , feature = None):     #[2, 15, 5000]
        if feature == None:
            l0_points = xyz
        else:
            l0_points = feature
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points, grouped_xyz, fps_idx = self.sa1(l0_xyz, l0_points)  
        # in [1, 3, 490]  [1, 16, 10000] out  [1, 3, 2048] [1, 128, 2048] [1, 2048, 64, 3] [1, 2048]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) 
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  

        # weight = self.fa(l4_points)
        # l4_points = weight*l4_points

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # o [2, 256, 64]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # o [2, 3, 1024]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # o [2, 128, 1024]

        # x = self.vote(l2_points)

        x = self.vote(l1_points)
        # x = F.sigmoid(x)
        return x, l1_xyz,fps_idx

def compute_displacement_map(mp,x2, k = 1):  # x1 npt > x2 npt
    x1 = mp.permute(0,2,1)
    # x2 = cp.permute(0,2,1)  #[b, np, ch]
    pairwise_distance = torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
    idx = pairwise_distance.topk(k=k, dim=2,largest=False)[1]     #[1, 10003, 1]
    # idx = idx.squeeze()
    idx = idx[...,0]
    target_point = index_points(x2, idx)
    offset = target_point - x1
    # x2 = x2.reshape(-1,x2.shape[2])
    # landmark =  x2[0][idx.squeeze()]  #.cpu().numpy()
    # s = x2[idx.squeeze()] 
    # offset =  landmark - x1
    return offset.permute(0,2,1)



if __name__ == '__main__':
    import os
    import vedo
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

    tample_path= r'J:\dataset\dental\CBCT_Mesh\1001055861_20180109_tample1.ply'
    v = vedo.load(tample_path).vertices
    v  = torch.from_numpy(v).unsqueeze(0)
    v = v.permute(0,2,1)

    face_path  =  r'J:\dataset\dental\CBCT_Mesh\tample_s490_npy\1000813648_20180116.npy'

    smaple = np.load(face_path, allow_pickle=True)  
    X = smaple.item()['X'] 
    mean_cell_centers = smaple.item()['mean_cell_centers']  
    max = smaple.item()['max'] 
    allp = smaple.item()['allp'] 

    X = torch.from_numpy(X).unsqueeze(0)
    print(X.shape)
    print(v.shape)

    pointnet = VoteNet()

    x, l1_xyz,fps_idx = pointnet(v,X)
    print(x, l1_xyz,fps_idx)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pointnet = VoteNet().cuda()
    # X = torch.rand(2,16,10000).cuda()
    # Y,xyz = pointnet(X)
    # print(Y.shape)
    # print(xyz)

    # Y_offset = torch.rand(2,3,10000).cuda()
    # Y_curve = torch.rand(2,700,3).cuda()
    # [x1,y1],[x2,y2] = pointnet(X,Y_offset,Y_curve)
    # print(y1.shape)
    # print(y2.shape)
    # print(y3.shape)
    # print(y4.shape)
    # summary(pointnet, (16, 10000))
