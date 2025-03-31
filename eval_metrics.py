import sys
import os
import torch
import argparse
import open3d as o3d
import numpy as np
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
        d1, _ = self.EMD(p1, p2, eps=0.005, iters=50)
        d = torch.sqrt(d1).mean(1).mean()
        return d


def read_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    return np.array(pcd.points)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--resdir', type=str, required=True, help='Full path to a single method results directory')
    parser.add_argument('--result_dir', type=str, required=True, help='Full path to output log file (e.g. ./logs/eval.txt)')
    args = parser.parse_args()

    dataset = args.dataset
    resdir = args.resdir
    result_file_path = args.result_dir

    # Extract method name from resdir path
    iname = os.path.basename(os.path.normpath(resdir))

    # Ensure result_dir's parent path exists
    dir_path = os.path.dirname(result_file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    sys.stdout = open(result_file_path, 'w')

    metrics = Metrics()
    father_dir = '/root/sfs/Data/zscomplete_data'
    gtdir = os.path.join(father_dir, dataset, 'gtdata')

    print(f"Evaluating method: {iname}")
    print(f"Dataset: {dataset}")
    print(f"Result path: {resdir}")

    objdict = {
        '01184': 'Trash Can', '06127': 'Plant in Vase', '06830': 'Tricycle', '07306': 'Office Trash Can',
        '05452': 'Outside Chair', '06145': 'One leg Table', '05117': 'Old Chair', '09639': 'Executive Chair',
        '06188': 'Vespa', '07136': 'Couch'
    }
    inlist = ['06145', '05452', '09639', '05117']
    outlist = ['06127', '07306', '06188', '06830']

    ipath = os.path.join(resdir, dataset)
    if not os.path.exists(ipath):
        print(f"Method path not found: {ipath}")
        sys.exit(1)

    filenames = os.listdir(os.path.join(father_dir, dataset, 'indata'))
    errs, inerrs, outerrs = [], [], []

    for ifile in filenames:
        shape_id = ifile.split('.')[0]
        if dataset == 'redwood' and shape_id not in inlist + outlist:
            continue

        print(f"Evaluating: {os.path.join(ipath, ifile)}")
        outdata = read_ply(os.path.join(ipath, ifile))
        gtdata = read_ply(os.path.join(gtdir, ifile))
        outdata = torch.tensor(outdata, dtype=torch.float32, device='cuda').unsqueeze(0)
        gtdata = torch.tensor(gtdata, dtype=torch.float32, device='cuda').unsqueeze(0)

        cd_l1 = metrics.chamfer_l1(outdata, gtdata).cpu().numpy()
        cd_l2 = metrics.chamfer_l2(outdata, gtdata).cpu().numpy()
        emd = metrics.emd_loss(outdata, gtdata).cpu().numpy()

        if dataset == 'redwood':
            print(objdict.get(shape_id, shape_id), ' cd1, cd2, emd: ', [cd_l1, cd_l2, emd])
            if shape_id in inlist:
                inerrs.append([cd_l1, cd_l2, emd])
            elif shape_id in outlist:
                outerrs.append([cd_l1, cd_l2, emd])
        else:
            print(shape_id, ' cd1, cd2, emd: ', [cd_l1, cd_l2, emd])
            errs.append([cd_l1, cd_l2, emd])

    # Print summary
    if dataset == 'redwood':
        inerrs = np.array(inerrs)
        outerrs = np.array(outerrs)
        errs = np.concatenate([inerrs, outerrs], axis=0) if inerrs.size and outerrs.size else np.array([])
        if errs.size > 0:
            print(f"{iname} Average cd1, cd2, emd: ", errs.mean(0))
            print("In domain average: ", inerrs.mean(0) if inerrs.size else "N/A")
            print("Out domain average: ", outerrs.mean(0) if outerrs.size else "N/A")
    else:
        errs = np.array(errs)
        if errs.size > 0:
            print(f"{iname} Average cd1, cd2, emd: ", errs.mean(0))

    sys.stdout.close()


if __name__ == '__main__':
    main()
