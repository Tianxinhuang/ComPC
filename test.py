from complete import *

import argparse
from omegaconf import OmegaConf
import random

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, default='configs/synthetic.yaml')
parser.add_argument("--workdir", required=False, default='')
parser.add_argument("--indir", required=False, default='')
parser.add_argument("--outdir", required=False, default='')
parser.add_argument("--name", required=False, default='plyobj')
params = parser.parse_args()

opt = OmegaConf.load(params.config)

def complete_one(workdir, indir, outdir, iname):

    torch.manual_seed(1024) #
    torch.cuda.manual_seed(1024) #
    np.random.seed(1024)
    random.seed(1024)
    torch.backends.cudnn.deterministic = True

    ipath = os.path.join(indir, iname)

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    gui = GUI(opt, workdir, indir, iname.split('.')[0])
    gui.train()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    os.system('cp -r ' +workdir+ '/eva.ply '+ os.path.join(outdir, gui.name+'_surf.ply'))
    os.system('cp -r ' +workdir+  '/gaussians.ply '+ os.path.join(outdir, gui.name+'_init.ply'))
    os.system('cp -r ' +workdir+  '/completed.ply '+ os.path.join(outdir, gui.name+'.ply'))
    os.system('cp -r ' +workdir+  '/depth.jpg '+ os.path.join(outdir, gui.name+'_depth.jpg'))
    os.system('cp -r ' +workdir+  '/render_rgba.png '+ os.path.join(outdir, gui.name+'_color.png'))

def complete_data(workdir, indir, outdir, dataname):
    indir = os.path.join(indir, dataname, 'indata')
    names = os.listdir(indir)
    
    for iname in names:
        complete_one(workdir, indir, os.path.join(outdir, dataname), iname)


if __name__=='__main__':
    workdir = params.workdir
    name = params.name
    #outdir = os.path.join(params.outdir, name)
    complete_data(workdir,params.indir, params.outdir, name)
