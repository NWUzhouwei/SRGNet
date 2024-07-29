#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: model.py
@Time: 2020/1/2 10:26 AM
"""

from typing import Dict
import torch
from torch import dropout
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import itertools
from util.loss import ChamferLoss, CrossEntropyLoss, TripletLoss


def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]            # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx


def local_cov(pts, idx):
    batch_size = pts.size(0)
    num_points = pts.size(2)
    pts = pts.view(batch_size, -1, num_points)              # (batch_size, 3, num_points)
 
    _, num_dims, _ = pts.size()

    x = pts.transpose(2, 1).contiguous()                    # (batch_size, num_points, 3)
    x = x.view(batch_size*num_points, -1)[idx, :]           # (batch_size*num_points*2, 3)
    x = x.view(batch_size, num_points, -1, num_dims)        # (batch_size, num_points, k, 3)

    x = torch.matmul(x[:,:,0].unsqueeze(3), x[:,:,1].unsqueeze(2))  # (batch_size, num_points, 3, 1) * (batch_size, num_points, 1, 3) -> (batch_size, num_points, 3, 3)
    x = x.view(batch_size, num_points, 9).transpose(2, 1)   # (batch_size, 9, num_points)
    x = torch.cat((pts, x), dim=1)                          # (batch_size, 12, num_points)

    return x


def local_maxpool(x, idx):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()                      # (batch_size, num_points, num_dims)
    x = x.view(batch_size*num_points, -1)[idx, :]           # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    x = x.view(batch_size, num_points, -1, num_dims)        # (batch_size, num_points, k, num_dims)
    x, _ = torch.max(x, dim=2)                              # (batch_size, num_points, num_dims)

    return x


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)      # (batch_size, num_dims, num_points)
    if idx is None:
        idx = knn(x, k=k)                       # (batch_size, num_points, k)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()          # (batch_size, num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[idx, :]                 # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)         # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (batch_size, num_points, k, num_dims)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)      # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)
  
    return feature                              # (batch_size, 2*num_dims, num_points, k)


class DGCNN_Cls_Encoder(nn.Module):
    def __init__(self, args):
        super(DGCNN_Cls_Encoder, self).__init__()
        if args.k == None:
            self.k = 40
        else:
            self.k = args.k
        self.task = args.task
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.feat_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(3*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.feat_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):        
        x = x.transpose(2, 1)

        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 512, num_points)

        x0 = self.conv5(x)                      # (batch_size, 512, num_points) -> (batch_size, feat_dims, num_points)
        x = x0.max(dim=-1, keepdim=False)[0]    # (batch_size, feat_dims, num_points) -> (batch_size, feat_dims)
        feat = x.unsqueeze(1)                   # (batch_size, feat_dims) -> (batch_size, 1, feat_dims)
        
        if self.task == 'reconstruct':
            return feat                         # (batch_size, 1, feat_dims)
        else:
            return feat, x0


class Point_Transform_Net(nn.Module):
    def __init__(self):
        super(Point_Transform_Net, self).__init__()
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x                                # (batch_size, 3, 3)


class DGCNN_Seg_Encoder(nn.Module):
    def __init__(self, args):
        super(DGCNN_Seg_Encoder, self).__init__()
        if args.k == None:
            self.k = 40
        else:
            self.k = args.k
        self.task = args.task
        self.transform_net = Point_Transform_Net()
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.feat_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.feat_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        

    def forward(self, x):
        x = x.transpose(2, 1)

        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x0 = self.conv6(x)                      # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)

        x = x0.max(dim=-1, keepdim=False)[0]    # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        feat = x.unsqueeze(1)                   # (batch_size, num_points) -> (batch_size, 1, emb_dims)

        if self.task == 'reconstruct':
            return feat                             # (batch_size, 1, emb_dims)
        elif self.task == 'classify':
            return feat, x0
        elif self.task == 'segment':
            return feat, x0, x1, x2, x3


class FoldNet_Encoder(nn.Module):
    def __init__(self, args):
        super(FoldNet_Encoder, self).__init__()
        if args.k == None:
            self.k = 16
        else:
            self.k = args.k
        self.n = 2048   # input point cloud size
        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, args.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, args.feat_dims, 1),
        )

    def graph_layer(self, x, idx):           
        x = local_maxpool(x, idx)    
        x = self.linear1(x)  
        x = x.transpose(2, 1)                                     
        x = F.relu(self.conv1(x))                            
        x = local_maxpool(x, idx)  
        x = self.linear2(x) 
        x = x.transpose(2, 1)                                   
        x = self.conv2(x)                       
        return x

    def forward(self, pts):
        pts = pts.transpose(2, 1)               # (batch_size, 3, num_points)
        idx = knn(pts, k=self.k)
        x = local_cov(pts, idx)                 # (batch_size, 3, num_points) -> (batch_size, 12, num_points])            
        x = self.mlp1(x)                        # (batch_size, 12, num_points) -> (batch_size, 64, num_points])    
        x = self.graph_layer(x, idx)            # (batch_size, 64, num_points) -> (batch_size, 1024, num_points)
        x = torch.max(x, 2, keepdim=True)[0]    # (batch_size, 1024, num_points) -> (batch_size, 1024, 1)
        x = self.mlp2(x)                        # (batch_size, 1024, 1) -> (batch_size, feat_dims, 1)
        feat = x.transpose(2,1)                 # (batch_size, feat_dims, 1) -> (batch_size, 1, feat_dims)
        return feat                             # (batch_size, 1, feat_dims)


class FoldNet_Decoder(nn.Module):
    def __init__(self, args):
        super(FoldNet_Decoder, self).__init__()
        self.m = 2025  # 45 * 45.
        self.shape = args.shape
        self.meshgrid = [[-0.3, 0.3, 45], [-0.3, 0.3, 45]]
        self.sphere = np.load("sphere.npy")
        self.gaussian = np.load("gaussian.npy")
        if self.shape == 'plane':
            self.folding1 = nn.Sequential(
                nn.Conv1d(args.feat_dims+2, args.feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(args.feat_dims, args.feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(args.feat_dims, 3, 1),
            )
        else:
            self.folding1 = nn.Sequential(
                nn.Conv1d(args.feat_dims+3, args.feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(args.feat_dims, args.feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(args.feat_dims, 3, 1),
            )  
        self.folding2 = nn.Sequential(
            nn.Conv1d(args.feat_dims+3, args.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, args.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, 3, 1),
        )

    def build_grid(self, batch_size):
        if self.shape == 'plane':
            x = np.linspace(*self.meshgrid[0])
            y = np.linspace(*self.meshgrid[1])
            points = np.array(list(itertools.product(x, y)))
        elif self.shape == 'sphere':
            points = self.sphere
        elif self.shape == 'gaussian':
            points = self.gaussian
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def forward(self, x):
        x = x.transpose(1, 2).repeat(1, 1, self.m)      # (batch_size, feat_dims, num_points)
        points = self.build_grid(x.shape[0]).transpose(1, 2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)
        if x.get_device() != -1:
            points = points.cuda(x.get_device())
        cat1 = torch.cat((x, points), dim=1)            # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
        folding_result1 = self.folding1(cat1)           # (batch_size, 3, num_points)
        cat2 = torch.cat((x, folding_result1), dim=1)   # (batch_size, 515, num_points)
        folding_result2 = self.folding2(cat2)           # (batch_size, 3, num_points)
        return folding_result2.transpose(1, 2)          # (batch_size, num_points ,3)


class DGCNN_Cls_Classifier(nn.Module):
    def __init__(self, args):
        super(DGCNN_Cls_Classifier, self).__init__()
        if args.dataset == 'modelnet40':
            output_channels = 40
        elif args.dataset == 'modelnet10':
            output_channels = 10
        elif args.dataset == 'shapenetcorev2':
            output_channels = 55
        elif args.dataset == 'shapenetpart':
            output_channels = 16
        elif args.dataset == 'shapenetpartpart':
            output_channels = 50

        self.linear1 = nn.Linear(args.feat_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class DGCNN_Seg_Segmenter(nn.Module):
    def __init__(self, args, seg_num_all):
        super(DGCNN_Seg_Segmenter, self).__init__()
        self.seg_num_all = seg_num_all
        self.seg_no_class_label = args.seg_no_class_label
        self.k = args.k
        self.feat_dims = args.feat_dims
        self.loss = args.loss
        
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        if not self.seg_no_class_label:
            self.conv8 = nn.Sequential(nn.Conv1d(self.feat_dims+64+64*3, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))
        else:
            self.conv8 = nn.Sequential(nn.Conv1d(self.feat_dims+64*3, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, x, l, x1, x2, x3):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        if not self.seg_no_class_label:
            l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
            l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

            x = torch.cat((x, l), dim=1)            # (batch_size, emb_dims+64, 1)
            x = x.repeat(1, 1, num_points)          # (batch_size, emb_dims+64, num_points)
            x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, emb_dims+64+64*3, num_points)

        else:
            x = x.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points)
            x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, emb_dims+64*3, num_points)

        x = self.conv8(x)                       # (batch_size, emb_dims+64+64*3 or emb_dims+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        
        if self.loss == 'softmax':
            x = self.conv11(x)                      # (batch_size, 128, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x


class ReconstructionNet(nn.Module):
    def __init__(self, args):
        super(ReconstructionNet, self).__init__()
        if args.encoder == 'foldnet':
            self.encoder = FoldNet_Encoder(args)
        elif args.encoder == 'dgcnn_cls':
            self.encoder = DGCNN_Cls_Encoder(args)
        elif args.encoder == 'dgcnn_seg':
            self.encoder = DGCNN_Seg_Encoder(args)
        self.decoder = FoldNet_Decoder(args)
        self.loss = ChamferLoss()

    def forward(self, input):
        feature = self.encoder(input)
        output = self.decoder(feature)
        return output, feature

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def get_loss(self, input, output):
        # input shape  (batch_size, 2048, 3)
        # output shape (batch_size, 2025, 3)
        return self.loss(input, output)


class ClassificationNet(nn.Module):
    def __init__(self, args):
        super(ClassificationNet, self).__init__()
        self.is_eval = args.eval
        if args.encoder == 'foldnet':
            self.encoder = FoldNet_Encoder(args)
        elif args.encoder == 'dgcnn_cls':
            self.encoder = DGCNN_Cls_Encoder(args)
        elif args.encoder == 'dgcnn_seg':
            self.encoder = DGCNN_Seg_Encoder(args)
        if not self.is_eval:
            self.classifier = DGCNN_Cls_Classifier(args)
        self.loss = CrossEntropyLoss()

    def forward(self, input):
        feature, latent = self.encoder(input)
        if not self.is_eval:
            output = self.classifier(latent)
            return output, feature
        else:
            return feature

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.classifier.parameters())

    def get_loss(self, preds, gts):
        # preds shape  (batch_size, feat_dims)
        # gts shape (batch_size)
        return self.loss(preds, gts)


class SegmentationNet(nn.Module):
    def __init__(self, args, seg_num_all=50):
        super(SegmentationNet, self).__init__()
        self.is_eval = args.eval
        self.loss_type = args.loss
        if args.encoder == 'foldnet':
            self.encoder = FoldNet_Encoder(args)
        elif args.encoder == 'dgcnn_cls':
            self.encoder = DGCNN_Cls_Encoder(args)
        elif args.encoder == 'dgcnn_seg':
            self.encoder = DGCNN_Seg_Encoder(args)
        if not self.is_eval:
            self.segmenter = DGCNN_Seg_Segmenter(args, seg_num_all)
        if self.loss_type == 'softmax':
            self.loss = CrossEntropyLoss()
        elif self.loss_type == 'triplet':
            self.loss = TripletLoss(margin=args.margin)

    def forward(self, input, label=None):
        feature, latent, x1, x2, x3 = self.encoder(input)
        if not self.is_eval:
            output = self.segmenter(latent, label, x1, x2, x3)
            return output, feature
        else:
            return feature

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.segmenter.parameters())

    def get_loss(self, preds, gts, new_device=None):
        # preds shape  (batch_size, feat_dims)
        # gts shape (batch_size)
        if self.loss_type == 'softmax':
            return self.loss(preds, gts)
        elif self.loss_type == 'triplet':
            return self.loss(preds, gts, new_device)

if __name__ == "__main__":
    ## knn
    # xyz = torch.rand(2, 3, 2048)
    # idx = knn(xyz, 3)
    # print(idx.shape)

    ## local_conv
    # xyz = torch.rand(2, 3, 2048)
    # idx = knn(xyz, 3)
    # result = local_cov(xyz, idx)
    # print(result.shape)

    ## local_maxpool
    # xyz = torch.rand(2, 3, 2048)
    # idx = knn(xyz, 3)
    # result = local_maxpool(xyz, idx)
    # print(result.shape)

    ## get_graph_feature
    # xyz = torch.rand(2, 3, 2048)
    # result = get_graph_feature(xyz)
    # print(result.shape)

    ## DGCNN_Cls_Encoder
    # xyz = torch.rand(2, 3, 2048)
    # xyz = xyz.transpose(2, 1)
    # class Args:
    #     k = 40
    #     task = None
    #     feat_dims = 20
    # args = Args
    # dgcnn_cls_encoder = DGCNN_Cls_Encoder(args)
    # result = dgcnn_cls_encoder(xyz)
    # print(result[0].shape, result[1].shape)

    ## Point_Transform_Net
    # xyz = torch.rand(2, 6, 2048, 3)
    # point_transform_net = Point_Transform_Net()
    # result = point_transform_net(xyz)
    # print(result.shape)

    ## DGCNN_Seg_Encoder
    # xyz = torch.rand(2, 3, 2048)
    # xyz = xyz.transpose(2, 1)
    # class Args:
    #     k = 40
    #     task = 'segment'
    #     feat_dims = 20
    # args = Args
    # dgcnn_seg_encoder = DGCNN_Seg_Encoder(args)
    # result = dgcnn_seg_encoder(xyz)
    # print(result[0].shape, result[1].shape, result[2].shape, result[3].shape, result[4].shape)

    ## FoldNet_Encoder
    # xyz = torch.rand(2, 3, 2048)
    # xyz = xyz.transpose(2, 1)
    # class Args:
    #     k = 16
    #     feat_dims = 20
    # args = Args
    # model = FoldNet_Encoder(args)
    # result = model(xyz)
    # print(result.shape) # (batch_size, 1, feat_dims)

    ## FoldNet_Decoder
    # xyz = torch.rand(2, 20, 1)
    # xyz = xyz.transpose(1, 2)
    # class Args:
    #     feat_dims = 20
    #     shape = 'sphere'
    # args = Args
    # model = FoldNet_Decoder(args)
    # result = model(xyz)
    # print(result.shape)

    ## DGCNN_Cls_Classifier
    # xyz = torch.rand(2, 3, 2048)
    # class Args:
    #     dataset = 'modelnet40'
    #     feat_dims = 3
    #     dropout = 0.1
    # args = Args
    # model = DGCNN_Cls_Classifier(args)
    # result = model(xyz)
    # print(result.shape)

    ## DGCNN_Seg_Segmenter
    # xyz = torch.rand(2, 3, 2048)
    # class Args:
    #     seg_no_class_label = True
    #     k = 3
    #     feat_dims = 3
    #     loss = 'softmax'
    #     dropout = 0.1
    # args = Args
    # model = DGCNN_Seg_Segmenter(args, 10)

    # l = torch.rand(2, 10)
    # x1 = torch.rand(2, 64, 2048)
    # x2 = torch.rand(2, 64, 2048)
    # x3 = torch.rand(2, 64, 2048)
    # result = model(xyz, l, x1, x2, x3)
    # print(result.shape)

    ## ReconstructionNet
    # xyz = torch.rand(2, 3, 2048)
    # xyz = xyz.transpose(2, 1)
    # class Args:
    #     k = 40
    #     task = 'reconstruct'
    #     feat_dims = 20
    #     encoder = 'foldnet'
    #     shape = 'sphere'        
    # args = Args
    # model = ReconstructionNet(args)
    # result = model(xyz)
    # print(result[0].shape, result[1].shape)

    ## ClassificationNet
    # xyz = torch.rand(2, 3, 2048)
    # xyz = xyz.transpose(2, 1)
    # class Args:
    #     k = 40
    #     task = 'classify'
    #     feat_dims = 20
    #     encoder = 'dgcnn_seg'
    #     shape = 'sphere'
    #     dataset = 'modelnet40'
    #     dropout = 0.1
    #     eval = True     
    # args = Args
    # model = ClassificationNet(args)
    # result = model(xyz)
    # print(result[0].shape, result[1].shape)
    # print(result.shape)

    ## SegmentationNet
    xyz = torch.rand(2, 3, 2048)
    xyz = xyz.transpose(2, 1)
    class Args:
        k = 40
        task = 'segment'
        feat_dims = 20
        encoder = 'dgcnn_seg'
        shape = 'sphere'
        dataset = 'modelnet40'
        dropout = 0.1
        eval = False
        loss = 'softmax'     
        seg_no_class_label = True
    args = Args
    model = SegmentationNet(args, seg_num_all=7)
    result = model(xyz)
    # print(result.shape)
    print(result[0].shape, result[1].shape)

    tmp = result[0].transpose(1, 2)
    print(tmp.shape)
    print(tmp[0][0])