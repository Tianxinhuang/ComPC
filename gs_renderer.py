import os
import math
import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement

import torch
from torch import nn
import torch.nn.functional as F

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from simple_knn._C import distCUDA2

from sh_utils import eval_sh, SH2RGB, RGB2SH
from mesh import Mesh
from mesh_utils import decimate_mesh, clean_mesh

import kiui
#from chamfer_distance import ChamferDistance
#from chamferdist import ChamferDistance
import sys
sys.path.append('ChamferDistancePytorch')
import chamfer3D.dist_chamfer_3D

from knn_cuda import KNN
#from pointnet2 import pointnet2_utils as pn2_utils
import sys
sys.path.append('pointnet2/pointnet2_ops_lib/pointnet2_ops')
import pointnet2_utils as pn2_utils
from pyhocon import ConfigFactory
from models.fields import NPullNetwork

import random
import open3d as o3d

def knn_point(group_size, point_cloud, query_cloud, transpose_mode=False):
    knn_obj = KNN(k=group_size, transpose_mode=transpose_mode)
    dist, idx = knn_obj(point_cloud, query_cloud)
    return dist, idx

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    
    def helper(step):
        if lr_init == lr_final:
            # constant lr, ignore other params
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10 # abnormal values... make weights 0
        
    return torch.exp(power)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            self.L = L
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.chamfer_dist = chamfer3D.dist_chamfer_3D.chamfer_3DDist()#ChamferDistance()
        self.sdf_network=None

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling.mean())*torch.ones_like(self.get_xyz)
    def neighbor_dist(self, pred):
        xyz = torch.cat([pred, self.raw_xyz],axis=0).unsqueeze(0)
        #print(xyz.shape, pred.shape)
        #assert False
        pred = pred.unsqueeze(0)
        _, idx = knn_point(2, xyz, pred, transpose_mode=True)
        #print(idx.shape)
        #assert False
        idx = idx[:, :, 1:].to(torch.int32) # remove first one
        idx = idx.contiguous() # B, N, nn

        xyz = xyz.transpose(1, 2).contiguous() # B, 3, N
        pred = pred.transpose(1, 2).contiguous()
        grouped_points = pn2_utils.grouping_operation(xyz, idx) # (B, 3, N), (B, N, nn) => (B, 3, N, nn)
        #print(grouped_points.shape)
        #assert False

        grouped_points = grouped_points - pred.unsqueeze(-1)
        dist2 = torch.sum(grouped_points ** 2, dim=1)
        #dist2 = torch.max(dist2, torch.tensor(1e-12).cuda())
        dist = torch.sqrt(dist2).squeeze(0)
        #print(dist.shape)
        #assert False
        return dist 
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
        #print(self._xyz.shape, self.raw_xyz.shape)
        #result = torch.cat([self._xyz,self.raw_xyz],axis=0)
        #print(result.shape)
        #assert False
        return result
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        #print(self._opacity.shape, self.get_xyz.shape)
        opa = self.opacity_activation(self._opacity)*torch.ones([self.get_xyz.shape[0],1]).cuda()
        opa = (torch.where(opa>=0.5, 1*torch.ones_like(opa), 0.01*torch.ones_like(opa))-opa).detach()+opa
        return opa

    def chamfer(self, inputs, points):
        dis1, dis2, _, _ = self.chamfer_dist(inputs, points)
        loss = ((1e-5+dis1).sqrt().mean() + (1e-5+dis2).sqrt().mean())/2
        return loss
    def merge(self, points, inputs):
        dis1, dis2, idx1, _ = self.chamfer_dist(points.unsqueeze(0), inputs.unsqueeze(0))
        #print(dis1.shape, idx1.shape)
        #assert False
        idx1 = idx1.squeeze(0).type(torch.long)
        move = (-dis1.detach().permute(1,0)/(1e-5+self.sigma.square())).exp()*(inputs[idx1.squeeze()]-points)
        newpoints = points + move
        movelen = move.square().sum(-1).sqrt().mean()
        return newpoints, movelen

    def select_attr(self, points, inputs):
        #print(points.shape, inputs.shape)
        dis1, dis2, idx1, idx2 = self.chamfer_dist(points.unsqueeze(0), inputs.unsqueeze(0))
        #print(idx1.shape, self._features_dc.shape)
        idx1 = idx1.squeeze(0).type(torch.long)
        #print(self._scaling.shape)

        colors = self._features_dc[idx1].detach().clone()
        #print(colors.shape)
        #assert False
        #colors_rest = self._features_rest[idx1].detach().clone()
        scales = self._scaling.detach().clone()
        #print(self._opacity.shape)
        opacity = self.get_opacity.detach().clone()#[idx1]
        rotation = self._rotation[idx1].detach().clone()
        return colors, scales, opacity, rotation

    def chamfer_sum(self, inputs, points):
        dis1, dis2, _, _ = self.chamfer_dist(inputs, points)
        loss = ((1e-5+dis1).sqrt().sum() + (1e-5+dis2).sqrt().sum())
        #loss = ((1e-5+dis1).sum() + (1e-5+dis2).sum())
        return loss
    def fidelity(self, inputs, points):
        #print(inputs.shape, points.shape)
        dis1, dis2, _, _ = self.chamfer_dist(inputs, points)
        loss = (1e-5+dis1).sqrt().mean() #+ (1e-5+dis2).sqrt().mean())/2
        return loss
    def extract_points(self):
        opacities = self.get_opacity
        #mask = (opacities > 0.005).squeeze(1)

        opacities = opacities#mask]
        xyzs = self.get_xyz#mask]
        stds = self.get_scaling#mask]

        ## normalize to ~ [-1, 1]
        #mn, mx = xyzs.amin(0), xyzs.amax(0)
        #self.center = (mn + mx) / 2
        #self.scale = 1.8 / (mx - mn).amax().item()

        #xyzs = (xyzs - self.center) * self.scale
        #stds = stds * self.scale

        covs = self.covariance_activation(stds, 1, self._rotation)
        std_gauss = torch.randn(xyzs.shape[0],xyzs.shape[1]).cuda()
        points = xyzs + opacities * (std_gauss.unsqueeze(1) @ self.L).squeeze(1)

        #cd = self.chamfer(points.unsqueeze(0), xyzs.unsqueeze(0))
        #print(cd)

        #print(self.L.shape,points.shape)

        return points

    def drive_sample(self, samples):
        #samples.requires_grad = True
        gradients_sample = self.sdf_network.gradient(samples).squeeze()
        grad_len = torch.linalg.norm(gradients_sample, dim=-1, keepdim=True)
        sdf_sample = self.sdf_network.sdf(samples)
        grad_norm = F.normalize(gradients_sample, dim=1)
        sample_moved = samples - grad_norm * sdf_sample
        self.drives = sample_moved
        return sample_moved, grad_len, sdf_sample

    #Select the points on the 3D grids near the surfaces
    def sdf_select(self, samples, interval, niter=3):
        offsets = [[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]
        offsets = torch.tensor(offsets, dtype=torch.float32, device='cuda')
        for i in range(niter):
            vsdf = self.sdf_network.sdf(samples).detach().clone().abs()
            radius = interval.square().sum().sqrt()/1.2
            idx = vsdf.squeeze() < radius
            samples = samples[idx]
            samples = samples.unsqueeze(1) + 0.6 * (interval * offsets).unsqueeze(0)
            samples = samples.reshape([-1,3])
            interval = interval / 2

        results = samples
        return results

    def set_sdf(self, signal):
        for params in self.sdf_network.parameters():
            params.requires_grad=signal
    def set_color(self, signal):
        for params in self.color_mlp.parameters():
            params.requires_grad=signal


    @torch.no_grad()
    def extract_fields(self, resolution=128, num_blocks=16, relax_ratio=1.5):
        # resolution: resolution of field
        
        block_size = 2 / num_blocks

        assert resolution % block_size == 0
        split_size = resolution // num_blocks

        opacities = self.get_opacity

        # pre-filter low opacity gaussians to save computation
        mask = (opacities > 0.005).squeeze(1)

        opacities = opacities[mask]
        xyzs = self.get_xyz[mask]
        stds = self.get_scaling[mask]
        #print(mask.shape, xyzs.shape)
        #assert False
        
        # normalize to ~ [-1, 1]
        mn, mx = xyzs.amin(0), xyzs.amax(0)
        self.center = (mn + mx) / 2
        self.scale = 1.8 / (mx - mn).amax().item()

        xyzs = (xyzs - self.center) * self.scale
        stds = stds * self.scale

        covs = self.covariance_activation(stds, 1, self._rotation[mask])

        # tile
        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)


        # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    # sample points [M, 3]
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                    # in-tile gaussians mask
                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                    # if hit no gaussian, continue to next block
                    if not mask.any():
                        continue
                    mask_xyzs = xyzs[mask] # [L, 3]
                    mask_covs = covs[mask] # [L, 6]
                    mask_opas = opacities[mask].view(1, -1) # [L, 1] --> [1, L]

                    # query per point-gaussian pair.
                    g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
                    g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

                    # batch on gaussian to avoid OOM
                    batch_g = 1024
                    val = 0
                    for start in range(0, g_covs.shape[1], batch_g):
                        end = min(start + batch_g, g_covs.shape[1])
                        w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
                        val += (mask_opas[:, start:end] * w).sum(-1)
                    
                    # kiui.lo(val, mask_opas, w)
                
                    occ[xi * split_size: xi * split_size + len(xs), 
                        yi * split_size: yi * split_size + len(ys), 
                        zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 
        
        kiui.lo(occ, verbose=1)

        return occ
    
    def extract_mesh(self, path, density_thresh=1, resolution=128, decimate_target=1e5):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        occ = self.extract_fields(resolution).detach().cpu().numpy()

        import mcubes
        vertices, triangles = mcubes.marching_cubes(occ, density_thresh)
        vertices = vertices / (resolution - 1.0) * 2 - 1

        # transform back to the original space
        vertices = vertices / self.scale + self.center.detach().cpu().numpy()

        vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.015)
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

        v = torch.from_numpy(vertices.astype(np.float32)).contiguous().cuda()
        f = torch.from_numpy(triangles.astype(np.int32)).contiguous().cuda()

        print(
            f"[INFO] marching cubes result: {v.shape} ({v.min().item()}-{v.max().item()}), {f.shape}"
        )

        mesh = Mesh(v=v, f=f, device='cuda')

        return mesh
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    def clip_dist(self, dist, c=3.0):
        meandist = dist.mean()
        dist = torch.clamp(dist, 0, c*meandist)
        return dist

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float = 1, z123 = False):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = 1*torch.clamp_min(distCUDA2(torch.from_numpy(np.concatenate([np.asarray(pcd.points)],axis=0)).float().cuda()), 0.0000001)
        all_scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        if z123:
            scales = 1.0*all_scales.mean(axis=1, keepdims=True).mean(axis=0,keepdims=True)[0]#.mean(axis=1,keepdims=True)
            opacities = inverse_sigmoid(0.40 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        else:
            scales = 1.1*all_scales.mean(axis=1, keepdims=True)#.mean(axis=0,keepdims=True)
            self.allscales = all_scales
            opacities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        sigma = torch.ones_like(scales, device="cuda")

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self.raw_xyz = torch.zeros_like(self._xyz)
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))#*torch.ones_like(all_scales)
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))#original is true
        self.sigma = nn.Parameter(sigma.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args, refine=False):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        if refine:
            self.del_tensor_in_optimizer("f_dc")

        else:     
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': self.sdf_network.parameters(), 'lr': 0.001, "name": "sdf"},
                {'params': self.sigma, 'lr': 0.001, "name": "sigma"},
            ]

            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        #self.sdf_optimizer = torch.optim.Adam(self.sdf_network.parameters(), lr=0.001, eps=1e-15)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        #l.append('scale')
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        #xyz = self._xyz.detach().cpu().numpy()
        xyz = self.get_xyz.cpu().numpy()
        #xyz = self.drives.cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()#.mean() #* np.ones([1,1])
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities * np.ones([xyz.shape[0],1]), scale * np.ones([xyz.shape[0],1]), rotation), axis=1)
        #print(attributes.shape, elements[:].shape)
        #print(list(map(tuple, attributes)))
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        self.spatial_lr_scale = 10.0
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        print("Number of points at loading : ", xyz.shape[0])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        #scales = np.asarray(plydata.elements[0]["scale"]).mean()
        scales = scales.mean()

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self.raw_xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))
        self.raw_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.raw_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        features_dc = np.concatenate([features_dc.repeat(self._xyz.shape[0]//self.snum, 0), features_dc[:self._xyz.shape[0] % self.snum]],axis=0)
        features_extra = np.concatenate([features_extra.repeat(self._xyz.shape[0]//self.snum, 0), features_extra[:self._xyz.shape[0] % self.snum]],axis=0)

        self.raw_feat = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False)

        #self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        #print(features_dc.max())
        #assert False

        self.raw_opacity = self.opacity_activation(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False).mean() * torch.ones_like(self.raw_xyz)[:,:1].cuda())
        self.raw_scaling = self.scaling_activation(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False).mean() * torch.ones_like(self.raw_xyz).cuda())
        self.raw_rotation = self.rotation_activation(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))

        #self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            #print(group['name'])
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    def selectpts(self):
        #print(self.get_opacity.shape)
        idx = self.get_opacity>=0.5
        #print(idx.sum())
        newxyz = self._xyz[idx.squeeze()]
        return newxyz, idx

    def del_tensor_in_optimizer(self, name):
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                #stored_state = self.optimizer.state.get(group['params'][0], None)
                #stored_state["exp_avg"] = torch.zeros_like(tensor)
                #stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                #print(group['name'])
                del self.optimizer.state[group['params'][0]]

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            #print(group['name'])
            if group['name'] not in self.dict_keys:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        #print(self._xyz.shape)

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        #self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        #print(self._xyz.shape)
        #assert False

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            #print(group['name'], tensors_dict.keys())
            #if group['name'] not in tensors_dict.keys():
            #    continue
            #assert len(group["params"]) == 1
            if group['name'] not in tensors_dict.keys():
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                #print(stored_state["exp_avg"].shape, extension_tensor.shape)

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        #"scaling" : new_scaling,
        "rotation" : new_rotation}
        self.dict_keys = d.keys()

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        #self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=4):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        #padded_grad = torch.zeros((n_init_points), device="cuda")
        #padded_grad[:grads.shape[0]] = grads.squeeze()
        #selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = self.get_opacity.squeeze(-1)>=0.5
        #selected_pts_mask = torch.logical_and(selected_pts_mask,
        #    torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent
        #)
        #xyz_selected_mask = selected_pts_mask[:-self.snum]
        #print(self._xyz.shape,  xyz_selected_mask.shape)
        #assert False

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        #new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        #print(samples.shape)
        #assert False
        new_xyz = self.get_xyz[selected_pts_mask].repeat(N, 1)+0.5*samples

        new_scaling = self._scaling#self.scaling_inverse_activation(self.get_scaling[selected_pts_mask,:3].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        #print(grads.shape)
        # Extract points that satisfy the gradient condition
        #selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = self.get_opacity.squeeze(-1)>=0.5
        #selected_pts_mask = torch.logical_and(selected_pts_mask,
        #    torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent
        #)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling#[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, N, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent, N)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        #if max_screen_size:
        #    big_points_vs = self.max_radii2D > max_screen_size
        #    big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        #    prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        #print(self._xyz, self.raw_xyz)

    def prune(self, min_opacity, extent, max_screen_size):

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        #if max_screen_size:
        #    big_points_vs = self.max_radii2D > max_screen_size
        #    big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        #    prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    #print(tanHalfFovY)
    #assert False

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1
        #w2c[1, 1] *= -1
        #print(w2c)
        #assert False

        #TR = np.array([[1,0,0,0],
        #      [0,-1,0,0],
        #      [0,0,-1,0],
        #       [0,0,0,1]]
        #      )
        #w2c = np.dot(c2w,TR) #
        #w2c = np.linalg.inv(w2c)
        #print(w2c)
        #assert False

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()
        #print(self.camera_center)
        #assert False


class Renderer:
    def __init__(self, sh_degree=3, white_background=True, radius=1):
        
        self.sh_degree = sh_degree
        self.white_background = white_background

        self.radius = radius

        self.gaussians = GaussianModel(sh_degree)

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
    
    def initialize(self, input=None, num_pts=5000, radius=0.5):
        # load checkpoint
        if input is None:
            # init from random point cloud
            num_pts=16384
            
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)
            # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            self.gaussians.create_from_pcd(pcd, 10)
            self.gaussians._xyz.requires_grad=True
            self.gaussians._scaling.requires_grad=False
            self.gaussians._opacity.requires_grad=False
            self.gaussians._rotation.requires_grad=False
        elif isinstance(input, BasicPointCloud):
            # load from a provided pcd
            self.gaussians.create_from_pcd(input, 1)
        else:
            # load from saved ply
            self.gaussians.load_ply(input)

    #Select points from 3D grid, and randomly select far points from the neighbor of bounding box
    def rand_sample(self, num_pts, surfpoints, out=False, boxsize=1.3):
        #Get some random noise
        xyz = np.random.rand(num_pts, 3)
        maxxyz2 = xyz.max(0)
        minxyz2 = xyz.min(0)
        maxxyz = surfpoints.max(0,keepdims=True)[0].cpu().numpy()
        minxyz = surfpoints.min(0,keepdims=True)[0].cpu().numpy()
        # Normalize the points by bounding box
        xyz = boxsize*(xyz-minxyz2)/(maxxyz2-minxyz2)*(maxxyz-minxyz)+minxyz
        # Increase the size of points a little to improve robustness 
        xyz = torch.tensor(np.asarray(xyz)).float().cuda()

        interval = torch.tensor(maxxyz-minxyz, dtype=torch.float32, device='cuda')/16
        #Get 3D grid with a specific resolution
        xyz0 = self.set_grid(minxyz[0], maxxyz[0], 16).cuda()
        if out:
            # Resample a uniform point clouds from a 3D grid through SDF func
            xyz0 = self.gaussians.sdf_select(xyz0, interval)
            xyz0 = self.fps_tensor(xyz0, 32768)# sample to 32768 points
        return xyz0, xyz

    #Get 3D grids
    def set_grid(self, bound_min, bound_max, resolution):
        N = resolution
        X = torch.linspace(bound_min[0]-0.0, bound_max[0]+0.0, resolution)
        Y = torch.linspace(bound_min[1]-0.0, bound_max[1]+0.0, resolution)
        Z = torch.linspace(bound_min[2]-0.0, bound_max[2]+0.0, resolution)
        xx, yy, zz = torch.meshgrid(X, Y, Z)
        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
        return pts

    #farthest point sampling
    def fps_tensor(self, data, num):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data.detach().cpu().numpy())
        pcd = pcd.farthest_point_down_sample(num)
        results =  torch.tensor(np.array(pcd.points), device='cuda', dtype=torch.float32)
        return results
    
    #Estimate the normal from point clouds
    def getnormal(self, data, pose):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)

        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Optionally orient the normals (assuming the sensor's viewpoint is known)
        pcd.orient_normals_towards_camera_location(camera_location=pose[:3,3])
        normal = np.array(pcd.normals)
        normal = (normal+1)/2
        return np.array(pcd.points), normal

    #define the colors by normal maps 
    def defcolor(self, pose):
        points, normals = self.getnormal(np.array(self.pcd.points), pose)
        pcd = BasicPointCloud(
                points=points, colors=normals, normals=np.zeros((normals.shape[0], 3)))

        self.gaussians.create_from_pcd(pcd, 10)

        self.gaussians._xyz.requires_grad=False
        self.gaussians._scaling.requires_grad=True
        self.gaussians._opacity.requires_grad=False
        self.gaussians._rotation.requires_grad=False


    def init2(self, path, scale_denom, z123=False):
        # Configuration
        self.conf_path = 'npull.conf'
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.gaussians.sdf_network = NPullNetwork(**self.conf['model.sdf_network']).to('cuda')
        self.gaussians.color_mlp = self.gaussians.sdf_network.trans3.to('cuda')

        self.scale_denom = scale_denom
        if z123:
            plydata = PlyData.read(path)

            xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                            np.asarray(plydata.elements[0]["y"]),
                            np.asarray(plydata.elements[0]["z"])),  axis=1)
            self.maxxyz = torch.tensor(xyz.max(0, keepdims=True)).float().cuda()
            self.minxyz = torch.tensor(xyz.min(0, keepdims=True)).float().cuda()

            rawxyz = xyz
            num_pts= 16384
            radius = 0.5

            self.gaussians.snum = xyz.shape[0]
            self.gaussians.raw_np = xyz

            kratio = 1.0
            ptnum = rawxyz.shape[0]
            kid = np.random.choice(list(range(ptnum)), size=int(kratio*ptnum), replace=False)
            xyz = rawxyz[kid] + 0.05*np.random.randn(len(kid),3)

            shs = np.random.random((num_pts, 3))
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )

            self.gaussians.create_from_pcd(pcd, 10, z123)
            self.gaussians.load_ply(path)

            self.gaussians._xyz.requires_grad=True
            self.gaussians._scaling.requires_grad=True
            self.gaussians._opacity.requires_grad=True
            self.gaussians._features_dc.requires_grad=True
            self.gaussians._features_rest.requires_grad=False
            self.gaussians._rotation.requires_grad=False
        else:
            plydata = PlyData.read(path)

            xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                            np.asarray(plydata.elements[0]["y"]),
                            np.asarray(plydata.elements[0]["z"])),  axis=1)
            num_pts = xyz.shape[0]

            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)

            shs = np.random.random((num_pts, 3)) / 255.0

            rgb = (xyz-xyz.min(0,keepdims=True))/(xyz.max(0, keepdims=True)-xyz.min(0, keepdims=True))
            pcd = BasicPointCloud(
                points=xyz, colors=rgb, normals=np.zeros((num_pts, 3))
            )
            self.pcd = pcd
            self.gaussians.create_from_pcd(pcd, 10)
            self.gaussians._xyz.requires_grad=False
            self.gaussians._scaling.requires_grad=True
            self.gaussians._opacity.requires_grad=False
            self.gaussians._rotation.requires_grad=False

    def normalize(self, data, cen=None, val=None):
        if cen is None or val is None:
            maxdata = data.max(0)[0]
            mindata = data.min(0)[0]
            #print(maxdata)
            #val = (maxdata-mindata).max()
            cen = (mindata+maxdata)*0.5
            #val = np.sqrt(np.square(data-cen).sum(-1,keepdims=True)).max()
            val = (maxdata-mindata).max()
        result = 0.5*(data-cen)/val
        return result, cen, val

    def render(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        fractal = False,
        manual_paras = None,
        surf = False,
        nfilter = None,
        pose=False,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass
        

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            if surf:
                scales = self.gaussians.neighbor_dist(self.gaussians.get_xyz.detach())*torch.ones_like(self.gaussians.get_xyz)
                #scales = 1*distCUDA2(torch.cat([means3D, self.gaussians.raw_xyz],dim=0))
                ##scales = 1*torch.clamp_min(distCUDA2(torch.cat([means3D],dim=0)), 0.0000001)
                #scales = scales.unsqueeze(-1)[:means3D.shape[0]]
                #scales = scales.sqrt() * torch.ones_like(means3D)
            else:
                scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

        if fractal:
            means3D = torch.cat([means3D, self.gaussians.raw_xyz],dim=0)
            means2D = torch.cat([means2D, torch.zeros_like(self.gaussians.raw_xyz)],dim=0)
            opacity = torch.cat([opacity, self.gaussians.raw_opacity],dim=0)
            scales = torch.cat([scales, self.gaussians.raw_scaling],dim=0)
            rotations = torch.cat([rotations, self.gaussians.raw_rotation],dim=0)


        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features
        else:
            colors_precomp = override_color

        if fractal:
            shs = torch.cat([shs,torch.cat([self.gaussians.raw_dc, self.gaussians.raw_rest],dim=1)],dim=0)

        if manual_paras is not None:
            means3D, shs, scales, opacity, rotations = manual_paras
            opacity = torch.ones_like(opacity)*torch.ones([means3D.shape[0],1]).cuda()
            scales = scales*torch.ones_like(means3D)

        if nfilter is not None:
            means3D = means3D[nfilter]
            means2D = means2D[nfilter]
            opacity = opacity[nfilter]
            scales = scales[nfilter]
            rotations = rotations[nfilter]

        if pose:
            scales = scales/self.scale_denom

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha, countlist = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": (radii > 0),#[:-self.gaussians.snum],
            "radii": radii,#[:-self.gaussians.snum],
            "surfcount": countlist,
        }
