import sys
import torch
import os
import open3d as o3d
import numpy as np

sys.path.append('..')
sys.path.append('external/SnowflakeNet/')

# from loss_functions import chamfer_l1, chamfer_l2, chamfer_partial_l1, chamfer_partial_l2, emd_loss
from loss_functions import chamfer_3DDist, emdModule

class Metrics:
    def __init__(self):
        self.chamfer_dist = chamfer_3DDist()
        self.EMD = torch.nn.DataParallel(emdModule().cuda()).cuda()

    def chamfer_l1(self, p1, p2):
        d1, d2, _, _ = self.chamfer_dist(p1, p2)
        d1 = torch.mean(torch.sqrt(d1))
        d2 = torch.mean(torch.sqrt(d2))
        return (d1 + d2) / 2

    def chamfer_l2(self, p1, p2):
        d1, d2, _, _ = self.chamfer_dist(p1, p2)
        return torch.mean(d1) + torch.mean(d2)

    def chamfer_partial_l1(self, pcd1, pcd2):
        d1, d2, _, _ = self.chamfer_dist(pcd1, pcd2)
        d1 = torch.mean(torch.sqrt(d1))
        return d1

    def chamfer_partial_l2(self, pcd1, pcd2):
        d1, d2, _, _ = self.chamfer_dist(pcd1, pcd2)
        d1 = torch.mean(d1)
        return d1

    def emd_loss(self, p1, p2):
        #print(p1.shape, p2.shape)
        d1, _ = self.EMD(p1, p2, eps=0.005, iters=50)
        d = torch.sqrt(d1).mean(1).mean()

        return d


def read_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    return np.array(pcd.points)



if __name__ == '__main__':
    metrics = Metrics()
    dataset = 'redwood'
    father_dir = '/root/sfs/Data/zscomplete_data'
    resdir = os.path.join(father_dir, 'results')
    gtdir = os.path.join(father_dir, dataset, 'gtdata')
    methods = os.listdir(resdir)
    print(methods)
    objdict={'01184':'Trash Can', '06127':'Plant in Vase', '06830':'Tricycle', '07306':'Office Trash Can', '05452':'Outside Chair', '06145':'One leg Table', '05117':'Old Chair', '09639':'Executive Chair', '06188': 'Vespa', '07136':'Couch'}
    inlist=['06145', '05452', '09639', '05117']
    outlist=['06127', '07306', '06188', '06830']
    for iname in methods:
        print(iname)
        ipath = os.path.join(resdir, iname, dataset)
        filenames = os.listdir(os.path.join(father_dir, dataset, 'indata'))
        errs=[]
        inerrs=[]
        outerrs=[]
        for ifile in filenames:
            if ifile.split('.')[0] not in inlist+outlist:
                continue
            print(os.path.join(ipath, ifile))
            outdata = read_ply(os.path.join(ipath, ifile))
            gtdata = read_ply(os.path.join(gtdir, ifile))
            outdata = torch.tensor(outdata, dtype=torch.float32, device='cuda').unsqueeze(0)
            gtdata = torch.tensor(gtdata, dtype=torch.float32, device='cuda').unsqueeze(0)
            cd_l1 = metrics.chamfer_l1(outdata, gtdata).cpu().numpy()
            cd_l2 = metrics.chamfer_l2(outdata, gtdata).cpu().numpy()
            emd = metrics.emd_loss(outdata, gtdata).cpu().numpy()
            print(objdict[ifile.split('.')[0]], ' cd1, cd2, emd, ', [cd_l1, cd_l2, emd])
            #print(ifile.split('.')[0], ' cd1, cd2, emd, ', [cd_l1, cd_l2, emd])
            
            #errs.append([cd_l1, cd_l2, emd])
            if ifile.split('.')[0] in inlist:
                inerrs.append([cd_l1, cd_l2, emd])
            elif ifile.split('.')[0] in outlist:
                outerrs.append([cd_l1, cd_l2, emd])

        errs = np.array(inerrs+outerrs)
        inerrs = np.array(inerrs)
        outerrs = np.array(outerrs)

        print(iname,'Average cd1, cd2, emd, ', errs.mean(0), '\n')
        print('In domain aver, ', inerrs.mean(0), '\n')
        print('Out domain aver, ', outerrs.mean(0), '\n')
