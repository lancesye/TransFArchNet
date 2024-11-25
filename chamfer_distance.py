import torch

def chamfer_distance(batch1, batch2):
    """
    使用torch.cdist计算两个批次3D Tensor之间的Chamfer距离。
    
    参数:
        batch1, batch2: 两个形状为[batch_size, num_points, 3]的Tensor，代表两批3D点云。
        
    返回:
        一个标量Tensor，代表两个批次点云之间的平均Chamfer距离。
    """
    # 使用torch.cdist计算两个批次中所有点对之间的距离
    batch1 = batch1.cuda()
    batch2 = batch2.cuda()
    pairwise_dist = torch.cdist(batch1, batch2, p=2)  # 结果形状为[batch_size, num_points_1, num_points_2]
    # 对于batch1中的每个点，找到batch2中最近的点，计算距离
    min_dist1, _ = torch.min(pairwise_dist, dim=2)  # 结果形状为[batch_size, num_points_1]
    # 对于batch2中的每个点，找到batch1中最近的点，计算距离
    min_dist2, _ = torch.min(pairwise_dist, dim=1) 
    # 计算Chamfer距离
    chamfer_dist = 0.5*torch.mean(min_dist1) + 0.5*torch.mean(min_dist2)
    return chamfer_dist,None # 返回所有批次的平均Chamfer距离