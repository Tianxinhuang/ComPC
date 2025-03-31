import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
#from diffusers import UNet2DModel
from torch.autograd import Variable
import sys
sys.path.append('/root/sfs/NeuralPull-Pytorch/models')
#from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
#import pointnet2_utils 

class ResidualBlock(torch.nn.Module):
    def __init__(self,channels=32):
        super(ResidualBlock,self).__init__()
        self.channels = channels

        self.conv1 = nn.Conv1d(channels, 64, 1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(64, channels, 1, stride=1, padding=0)

    def forward(self, x):
        #x = x.unsqueeze(-1)
        #print(x.shape)
        #assert False
        #print(x.shape)
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        #y = y.squeeze(-1)
        #x = x.squeeze(-1)
        #print(x.shape, y.shape)
        #assert False
        #print(x.shape,y.shape)
        return x+y

class NPullNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(NPullNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        #print(dims)
        #assert False

        self.embed_fn_fine = None

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)
            #res = ResidualBlock(out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
            #setattr(self, "res" + str(l), res)
            
        self.activation = nn.ReLU()#nn.Softplus(beta=100)
        self.dropout = nn.Dropout(p = 0.5)
        #self.activation = nn.Softplus(beta=10)

        


        self.prim = 8
        self.dnum = 16
        self.cen_dim = 1024
        #self.knum = 128
        #self.para = 0.1*torch.ones(1, dtype=torch.float32, device='cuda')
        #self.para.requires_grad = True
        #self.feats = torch.rand([self.prim, self.dnum], dtype=torch.float32, device='cuda')
        #self.feats.requires_grad = True

        #self.reso = 8
        #self.feat_grid = torch.rand([3, self.reso, self.reso, self.dnum], dtype=torch.float32, device='cuda')
        #self.feat_grid.requires_grad = True

        #xcoor=torch.tensor([list(range(self.reso))]*self.reso,dtype=torch.float32).to('cuda')
        #ycoor=torch.tensor([list(range(self.reso))]*self.reso,dtype=torch.float32).to('cuda')
        #ycoor=ycoor.transpose(0,1)
        #coors=torch.cat([xcoor.unsqueeze(-1),ycoor.unsqueeze(-1)],dim=-1)
        #self.coor_grid=self.reso/(self.reso-1) * coors.unsqueeze(0).repeat((3,1,1,1))/(self.reso*1.0)
        
        #print(self.coor_grid.shape)#1,32,32,2
        self.wei = torch.Tensor(self.cen_dim).unsqueeze(0)
        nn.init.normal_(self.wei, 0, 0.1)
        self.wei.requires_grad=True
        self.centers=None
        self.outlayer = nn.Conv1d(self.cen_dim, 1, 1, stride=1, padding=0)

        self.trans = nn.Sequential(
            nn.Conv1d(3 , 32, 1, stride=1, padding=0),
            nn.Softplus(),
            nn.Conv1d(32, 64, 1, stride=1, padding=0),
            nn.Softplus(),
            nn.Conv1d(64, 256, 1, stride=1, padding=0),
            nn.Softplus(),
            nn.Conv1d(256, 256, 1, stride=1, padding=0),
            nn.Softplus(),
            #nn.Conv1d(256, 256, 1, stride=1, padding=0),
            )
        self.trans2 = nn.Sequential(
            nn.Conv1d(3, 16, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(16, 64, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1, stride=1, padding=0),
            #nn.Tanh()
            )

        self.trans3 = nn.Sequential(
            nn.Conv1d(3, 16, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(16, 64, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1, stride=1, padding=0),
            #nn.Tanh()
            )
        self.rsblocks = nn.Sequential(
            nn.Conv1d(3, 256, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1, stride=1, padding=0),
            nn.ReLU(),
            #nn.Conv1d(256, 256, 1, stride=1, padding=0),
            #nn.ReLU(),
            #nn.Conv1d(256, 256, 1, stride=1, padding=0),
            #nn.ReLU(),
            #nn.Conv1d(256, 256, 1, stride=1, padding=0),
            #nn.ReLU(),
            #ResidualBlock(256),
            #ResidualBlock(256),
            #ResidualBlock(256),
            nn.Conv1d(256, 1, 1, stride=1, padding=0)
            #nn.Tanh()
            )
        self.minbound = None
        self.maxbound = None
        #self.group = pointnet2_utils.QueryAndGroup(0.4,self.knum, use_xyz=False)
        #self.qpoints = None

        #self.SA_modules = nn.ModuleList()
        #self.SA_modules.append(
        #self.msg = PointnetSAModuleMSG(
        #        npoint=-1,
        #        radii=[0.2],
        #        nsamples=[64],
        #        mlps=[[3, 64, 64, 256]],
        #        use_xyz=False,
        #    )
        

    def forward(self, inputs):
        inputs = inputs * self.scale
        #inputs = inputs.unsqueeze(0)#.permute(0,2,1).contiguous()
        ##new_xyz = (pointnet2_utils.gather_operation(qpoints, pointnet2_utils.furthest_point_sample(qpoints, self.npoint)).transpose(1, 2).contiguous())
        #nfeat = self.group(self.qpoints.unsqueeze(0), inputs, self.qpoints.unsqueeze(0).permute(0,2,1).contiguous())
        #nfeat = nfeat.squeeze(0).permute(1,0,2)
        #nfeat = nfeat - inputs.permute(1,2,0)
        #x = self.trans(nfeat).squeeze(1)

        #print(x.shape)
        #assert False
        #newxyz, msgfeat = self.msg(inputs, features=inputs.permute(0,2,1).contiguous())
        #lin = getattr(self, "lin" + str(self.num_layers - 2))
        #x = lin(msgfeat.squeeze(0).permute(1,0))
        #print(x.shape)
        #print(msgfeat.shape)
        #assert False
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)
        #print(inputs.shape)
        #x = self.rsblocks(inputs.unsqueeze(-1)).squeeze(-1)
        #x = self.svm(self.centers, inputs)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            #res = getattr(self, "res" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            #if l==0 or l==self.num_layers - 2:
            x = lin(x)
            #if l==0:
            #x = self.activation(x)
            #else:
                #print(x.shape)
            #    x = res(x)
            #if l==0:
            #    x = self.activation(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
            #if l == 1 or l==0:
            #    x = res(x)
        #    print(x.shape, self.num_layers)
        ##assert False
        #    #    x = self.dropout(x)
        #    #if l == self.num_layers - 3:
        #    #    x = self.activation(x)
        #        #x = torch.sin(30 * x)
        ##print(x.shape)
        ##assert False
        ##x = x.min(-1,keepdim=True)[0]
        return x / self.scale

    #def forward(self, inputs):
    #    inputs = inputs * self.scale
    #    #print(inputs.shape, self.minbound)
    #    #assert False
    #    inputs = (inputs-self.minbound)/(self.maxbound-self.minbound)
    #    #print(self.coor_grid.min(), self.coor_grid.max())
    #    #assert False
    #    xz = torch.cat([inputs[:,:1], inputs[:,2:]],axis=-1).unsqueeze(0)
    #    xy = inputs[:,:2].unsqueeze(0)
    #    yz = inputs[:,1:].unsqueeze(0)
    #    
    #    xyz = torch.cat([xy, yz, xz],axis=0)
    #    
    #    pnum = inputs.shape[0]
    #    dis = (xyz.unsqueeze(2)-self.coor_grid.reshape([3, -1, 2]).unsqueeze(1)).abs().sum(-1,keepdim=True)
    #    #print(dis.min(),dis.max())
    #    #assert False

    #    weight = 1*(-dis).exp()#/(-dis).exp().sum(2,keepdim=True)
    #    #print(weight.min(),weight.max())
    #    #print(dis.shape, self.feat_grid.reshape([3, 1, -1, self.dnum]).shape)
    #    #assert False
    #    feats = weight * self.feat_grid.reshape([3, -1, self.dnum]).unsqueeze(1).repeat(1, pnum, 1, 1)
    #    feats = feats.sum(2)
    #    
    #    x=feats.permute(1,2,0)
    #    x=x.reshape([pnum,-1, 1])
    #    x=self.trans(x).squeeze(-1)
    #    #print(x.shape)
    #    #assert False

    #    #x=(1e-8+x.square().sum(-1,keepdim=True)).sqrt()


    #    return x / self.scale

    def sdf(self, x):
        return self.forward(x)
    def sdftrans(self, x):
        x = x.unsqueeze(0)
        x= x.permute(0,2,1)
        res=self.trans(x).permute(0,2,1)
        res=res.squeeze(0)
        rep = torch.sigmoid(res)
        #rep = self.para.square() + (-res.square()).exp()
        #res = rep/(1e-12+rep.sum())
        #res = res.exp()/(1e-6+res.exp().sum())
        #paraloss = (-self.para.square().mean()).exp()
        paraloss = 0.001* (1-rep.mean()) + torch.relu(0.9-rep.mean())
        return rep, paraloss
    def sdftrans2(self, x):
        x = x.unsqueeze(0)
        x= x.permute(0,2,1)
        res=self.trans2(x).permute(0,2,1)
        res=res.squeeze(0)
        return res
    def sdftrans3(self, x):
        x = x.unsqueeze(0)
        x= x.permute(0,2,1)
        res=self.trans3(x).permute(0,2,1)
        res=res.squeeze(0)
        return res
    ##centers:M*3, x:N*3
    #def svm(self, centers, x):
    #    cens = centers.unsqueeze(-1)#.permute(0,2,1)
    #    cens = cens + self.trans3(cens)
    #    #print(cens.shape)
    #    #cens = cens.permute(2,1,0)
    #    cenfeats = self.trans(cens).squeeze(-1)#M*d*1
    #    #print(cenfeats)
    #    xfeats = self.trans(x.unsqueeze(-1)).squeeze(-1)#N*d*1
    #    #print(xfeats.shape, cenfeats.shape)
    #    #assert False
    #    #cross = (xfeats.)
    #    dis = (x.unsqueeze(1)-centers.unsqueeze(0)).square().sum(-1)
    #    #print(dis.shape)
    #    #assert False
    #    cross = torch.matmul(xfeats, cenfeats.permute(1,0))*(-dis).exp()#N*M
    #    #cross = (-dis).exp()
    #    #wei = torch.inverse(torch.matmul(cross.permute(1,0),cross)).matmul()
    #    #result = torch.matmul(cross, self.wei.permute(1,0))
    #    result = self.outlayer(cross.unsqueeze(-1)).squeeze(-1)
    #    return result


    #def folding(self, num,  centers):
    #    cens = centers.permute(1,2,0)#.permute(0,2,1)
    #    #cens = cens #+ self.trans3(cens)
    #    #cens = cens.permute(2,1,0)

    #    pnum = num//self.prim
    #    rand_grid = Variable(torch.cuda.FloatTensor(1 ,pnum, 2, 1))
    #    rand_grid.data.uniform_(0,1)
    #    rand_grid = rand_grid.repeat([self.prim, 1, 1, 1])
    #    #print(self.feats.shape)
    #    #feats = torch.cat([rand_grid, self.feats[:,None, :, None].repeat([1,pnum,1,1])],axis=2)
    #    #assert False
    #    #print(cens.shape, rand_grid.shape)
    #    #assert False
    #    #feats = torch.cat([rand_grid, cens.unsqueeze(1).repeat([1,pnum,1,1]).detach()],axis=2)
    #    feats = torch.cat([rand_grid],axis=2)
    #    self.dnum=3
    #    #print(feats.shape)
    #    #assert False
    #    feats = feats.permute(0,2,1,3).squeeze()
    #    #feats = feats.reshape([self.prim,self.dnum+2, pnum])
    #    #print(feats.shape)
    #    #assert False
    #    points = self.trans3(self.trans2(feats)) #+ cens
    #    points = points.permute(0,2,1)
    #    #print(points.shape)
    #    #assert False
    #    points = points.reshape([-1,3])
    #    #assert False
    #    #feats = torch.cat([points, feats])
    #    return points
    #def difftrans(self,x,t):
    #    sample = self.trans(x)
    #    timesteps = t
    #    timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)
    #    t_emb = self.diff.time_proj(timesteps)
    #    # timesteps does not contain any weights and will always return f32 tensors
    #    # but time_embedding might actually be running in fp16. so we need to cast here.
    #    # there might be better ways to encapsulate this.
    #    t_emb = t_emb.to(dtype=self.dtype)
    #    emb = self.diff.time_embedding(t_emb)
    #    sample = self.diff.mid_block(sample, emb)

    #    sample = self.diff.conv_norm_out(sample)
    #    sample = self.diff.conv_act(sample)
    #    sample = self.diff.conv_out(sample)
    #    return sample



    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        # y.requires_grad_(True)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)
    def gradient2(self, x):
        #grad1 = self.gradient(x)
        x.requires_grad_(True)
        y = self.sdf(x)
        # y.requires_grad_(True)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        grad1 = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=False)[0]

        d_output = torch.ones_like(x, requires_grad=False, device=y.device)
        #gradients = torch.autograd.backward(grad1, grad_tensors=d_output,
        #   create_graph=True, retain_graph=True)
        #print(grad1.shape)
        #assert False
        gradients = torch.autograd.grad(
            outputs=grad1,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        #print(grad1, gradients)
        #assert False
        return grad1.unsqueeze(1), gradients.unsqueeze(1)

#class RBF(nn.module):
#    def __init__(self, cen_num=64, dim=3):
#        self.cennum=cen_num
#        self.dim=dim
#        self.cenpara=nn.Parameter(torch.Tensor(self.cennum, self.dim))
#        self.cval=nn.Parameter(torch.Tensor(out_features))






def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    Suggested by https://github.com/mikedh/trimesh/issues/507
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        print("is_mesh")
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh
