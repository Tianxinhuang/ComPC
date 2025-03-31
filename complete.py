import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F

import rembg

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize
from torchvision import utils as vutils

import requests
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import open3d as o3d
import random

def write_ply_ascii_geo(filedir, coords):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    f.write('end_header\n')
    coords2 = np.array(coords)
    for i in range(len(coords2)):
        p = coords[i]
        try:
            f.writelines([str(p[0]), ' ', str(p[1]), ' ',str(p[2]), '\n'])
        except:
            continue
    f.close()

    return

class GUI:
    def __init__(self, opt, zsdir, plydir, name):

        torch.manual_seed(1024) #
        torch.cuda.manual_seed(1024) #
        np.random.seed(1024)
        random.seed(1024)
        torch.backends.cudnn.deterministic = True

        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        #self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop

        self.zsdir = zsdir
        self.posedir = os.path.join(self.zsdir, 'pose.npy')
        self.plydir = plydir
        self.name = name

        self.pts_path = os.path.join(self.plydir, self.name+'.ply')
        self.fov = opt.fovy/180*np.pi
        self.relu = torch.nn.ReLU(inplace=True)

        self.densify_N = opt.densify_N
        self.pose_w = opt.pose_weight
        self.scale_w = opt.scale_weight
        self.box_size = opt.box_size

        if not hasattr(self, 'angles'):
            self.intri = [1024, 1024, 100]
            self.extri = None
            self.angles = None

        controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
        #
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
        
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt

        # override if provide a checkpoint
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load)            
        else:
            # initialize gaussians to a blob
            self.renderer.init2(path=self.pts_path, scale_denom=opt.scale_denom)
            self.points = self.renderer.gaussians._xyz.detach()

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        pose = orbit_camera(self.angles[0], self.angles[1], 1)
        fov = self.fov#2*np.arctan(self.intri[0]/(2*self.intri[2]))/np.pi * 180
        self.fixed_cam = MiniCam(pose, self.intri[0],self.intri[1], fov, fov, 0.01, 100)

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            elif self.opt.raw_sd:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")
            elif self.opt.depth_sd:
                print(f"[INFO] loading SD...")
                from guidance.dp_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")
            else:
                print(f"[INFO] loading controlnet SD...")
                from guidance.ct_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded controlnet SD!")

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            self.guidance_zero123 = Zero123(self.device)
            #print(f"[INFO] loaded zero123!")
            if self.opt.stable_zero123:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/stable-zero123-diffusers')
                #self.guidance_zero123 = Zero123(self.device)
            else:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')

        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img = self.input_img_torch
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask = self.input_mask_torch
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)
    import cv2

    #find a pose that most points can be observed, also near the points
    def pose_step(self):
        bestangles = None
        poses = []
        angles = []
        loss = []

        startv, endv = -80,80
        starth, endh = -180, 180
        num = 50
        points = None

        for j in range(2):

            vers = torch.linspace(float(startv), float(endv), num)
            hors = torch.linspace(float(starth), float(endh), num)
            verss, horss = torch.meshgrid(vers, hors)
            verss = verss.reshape([-1])
            horss = horss.reshape([-1])

            loss = []
            poses = []
            angles = []
            pointlist=[]

            for i in range(int(num*num)):
                #ver = np.random.randint(-60, 60)
                #hor = np.random.randint(-180, 180)
                ver = verss[i].cpu().numpy()
                hor = horss[i].cpu().numpy()

                pose = orbit_camera(ver, hor, 1)
                #pose = np.dot(pose, np.array(self.extri, dtype=np.float32))
                poses.append(pose)
                angles.append([ver,hor])

                fov = self.fov#self.fov#2*np.arctan(self.intri[0]/(2*self.intri[2]))/np.pi * 180
                #cur_cam = MiniCam(pose, self.intri[0], self.intri[1], fov, fov, 0.01, 100)
                cur_cam = MiniCam(pose, self.intri[0], self.intri[1], fov, fov, 0.001, 1000)

                out = self.renderer.render(cur_cam, surf=False, pose=True)
                #image = out["image"]
                fixed_countlist = out['surfcount']>0
                #xyz = torch.cat([self.renderer.gaussians._xyz, self.renderer.gaussians.raw_xyz], dim=0)
                xyz = self.renderer.gaussians._xyz
                #xyz, cen, val = self.renderer.normalize(xyz)
                fixed_points = xyz[fixed_countlist]
                pointlist.append(fixed_points.clone())
                #points = fixed_points
                #print(fixed_points.shape, xyz.shape)
                #assert False
                fixed_cd = self.renderer.gaussians.fidelity(xyz.unsqueeze(0),fixed_points.unsqueeze(0))
                cen = torch.tensor(pose[:3,3], device='cuda')
                posedist = (cen.unsqueeze(0)-xyz).square().sum(-1).sqrt().mean()

                lossi = fixed_cd + self.pose_w * posedist #+ 0.0001*abs(ver)
                loss.append(lossi.cpu().numpy())

            minid = np.argmin(loss)
            self.angles = angles[minid]
            self.extri = poses[minid]
            interv = (endv-startv)/num
            interh = (endh-starth)/num

            startv, endv = verss[minid].cpu().numpy()-interv, verss[minid].cpu().numpy()+interv
            starth, endh = horss[minid].cpu().numpy()-interh, horss[minid].cpu().numpy()+interh

    def depth_process(self, depth, mask):
        depth = depth.squeeze().unsqueeze(-1)
        mask  = mask.squeeze().unsqueeze(-1)
        maxdepth = (mask*depth).max()
        mindepth = (mask*depth+(1-mask)*torch.ones_like(mask)).min()
        depth = (depth-mindepth)/(maxdepth-mindepth) #* mask
        #print(depth.shape)
        depth = (1-depth)*mask
        depth = depth.permute(2,0,1)
        return depth.detach()
    
    def color_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        images=[]
        ver, hor = 0.0,0.0

        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0

            cur_cam = self.fixed_cam
            out = self.renderer.render(cur_cam)
            image = out["image"]
            if self.step == 1:
                self.image = image.detach()
            depth = self.depth_process(out['depth'],  out['alpha'])
            loss = loss + 10000 * (out["alpha"].unsqueeze(0)*(image-self.image)).abs().mean()

            ### novel view (manual batch)
            #render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)

            for _ in range(self.opt.batch_size):

                #pose = np.array(self.extri, dtype=np.float32) ##data camera pose by extrinsic
                pose = orbit_camera(self.angles[0]+ver, self.angles[1]+hor, 1)

                #poses.append(pose)
                fov = self.fov#2*np.arctan(self.intri[0]/(2*self.intri[2]))/np.pi * 180
                cur_cam = MiniCam(pose, self.intri[0], self.intri[1], fov, fov, 0.01, 100)

                if self.step > 4:
                    bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
                else:
                    bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.renderer.render(cur_cam, bg_color=bg_color)

                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)

            images = torch.cat(images, dim=0)
            scaling = self.renderer.gaussians.get_scaling
            scale_loss = 3000*scaling.mean()
            opa_loss = 100*(self.renderer.gaussians.get_opacity).abs().mean()

            # guidance loss
            if self.enable_sd:
                if self.opt.raw_sd:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio=step_ratio if self.opt.anneal_timestep else None)
                elif self.opt.depth_sd:
                    depth = self.depth_process(out['depth'],  out['alpha'])
                else:
                    depth =  self.depth_process(out['depth'],  out['alpha'])#(out['depth'].max()-out['depth']) * out['alpha']

            loss += 1.0 * scale_loss

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.save_image(images, self.zsdir+'/render.jpg')
        self.save_image(depth.unsqueeze(0), self.zsdir+'/depth.jpg')

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

    #def color_step(self):
    #    images=[]

    #    #for _ in range(self.train_steps):

    #    #self.step = 1
    #    #step_ratio = min(1, self.step / self.opt.iters)

    #    ## update lr
    #    #self.renderer.gaussians.update_learning_rate(self.step)

    #    loss = 0

    #    cur_cam = self.fixed_cam
    #    out = self.renderer.render(cur_cam)
    #    image = out["image"]
    #    self.image = image.detach()
    #    depth = self.depth_process(out['depth'],  out['alpha'])
    #    #loss = loss + 10000 * (out["alpha"].unsqueeze(0)*(image-self.image)).abs().mean()

    #    #for _ in range(self.opt.batch_size):

    #    pose = orbit_camera(self.angles[0], self.angles[1], 1)

    #    fov = self.fov#
    #    cur_cam = MiniCam(pose, self.intri[0], self.intri[1], fov, fov, 0.01, 100)

    #    bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    #    out = self.renderer.render(cur_cam, bg_color=bg_color)

    #    image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
    #    images.append(image)

    #    images = torch.cat(images, dim=0)

    #    self.save_image(images, self.zsdir+'/render.jpg')
    #    self.save_image(depth.unsqueeze(0), self.zsdir+'/depth.jpg')

    #    torch.cuda.synchronize()

    #itera: iteration id
    def train_step(self, itera):

        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0

            ### known view
            if self.input_img_torch is not None:
                cur_cam = self.fixed_cam
                self.refined, self.idx = self.renderer.gaussians.selectpts()
                noise_filter = torch.cat([self.idx.squeeze(), torch.ones(self.renderer.gaussians.raw_xyz.shape[0]).cuda()], dim=0)

                out = self.renderer.render(cur_cam, fractal=True)

                # rgb loss
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                loss = loss + 10000 * step_ratio * ((image-self.input_img)).abs().mean()

                # mask loss
                mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
                loss = loss + 10000 * step_ratio * (mask-self.input_mask).abs().mean()

                # Acquire the 
                fixed_countlist = out['surfcount'] > 0
                xyz = torch.cat([self.renderer.gaussians._xyz, self.renderer.gaussians.raw_xyz], dim=0)
                fixed_points = xyz[fixed_countlist]

                # Preservation Constraint
                fixed_cd =  100 * self.renderer.gaussians.chamfer_sum(fixed_points.unsqueeze(0), self.renderer.gaussians.raw_xyz.unsqueeze(0)) 
                loss = loss + fixed_cd

            ### novel view (manual batch)
            render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            images = []
            poses = []
            vers, hors, radii = [], [], []

            samples_list=[]
            points_list=[]

            vers = torch.linspace(-self.opt.ver_mean, self.opt.ver_mean, 4)
            hors = torch.linspace(-self.opt.hor_mean, self.opt.hor_mean, 4)

            verss, horss = torch.meshgrid(vers, hors)
            verss = verss.reshape([-1])
            horss = horss.reshape([-1])

            vers = verss + self.opt.ver_std * torch.randn(size=verss.shape)
            hors = horss + self.opt.hor_std * torch.randn(size=horss.shape)

            vers = vers.cpu().numpy()
            hors = hors.cpu().numpy()

            for bs in range(16):

                # render random view
                ver = vers[bs]
                hor = hors[bs]
                radius = 0

                radii.append(radius)

                pose = orbit_camera(ver, hor, 1)
                poses.append(pose)

                fov = self.fov
                cur_cam = MiniCam(pose, self.intri[0], self.intri[1], fov, fov, 0.01, 100)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.renderer.render(cur_cam, bg_color=bg_color, fractal=True)

                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)
                
            images = torch.cat(images, dim=0)

            scaling = self.renderer.gaussians.get_scaling
            scale_loss = self.scale_w * scaling.square().mean()

            col_loss = 100 * self.renderer.gaussians.chamfer_sum(self.renderer.gaussians._features_dc.permute(1,0,2), self.renderer.gaussians.raw_feat.permute(1,0,2).detach())

            # View dependent guidance
            if self.enable_zero123:
                loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers-self.angles[0], hors-self.angles[1], radii, step_ratio=step_ratio if self.opt.anneal_timestep else None)

            loss += scale_loss
            #loss += col_loss
            
            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if (self.step+1) % self.opt.densification_interval == 0:
                print('densify...')
                self.renderer.gaussians.densify_and_prune(0.0, min_opacity=0.5, extent=4, N = self.densify_N, max_screen_size=1)

        self.save_image(images[-1:], self.zsdir+'/render.jpg')
        torch.cuda.synchronize()
    
    #itera: iteration id, itere: iteration end
    def refine_step(self, itera, itere = 3000):
        self.renderer.gaussians.set_sdf(True)
        self.renderer.gaussians.sdf_network.train()
        samnum = self.surfpoints.shape[0]
        
        out=False
        if itera >= itere-1:
            out= True

        #Randomly sample the grids, points 
        grids, samples_far, samples_near = self.rand_sample(self.surfpoints, samnum=16384, uniform=False, out=out, boxsize = self.box_size)

        #Use SDF to drive far points
        points_coarse, grad_len, movelen = self.renderer.gaussians.drive_sample(samples_far)

        #Use SDF to drive near points
        points_fine, _, _ = self.renderer.gaussians.drive_sample(samples_near)

        #Merge the driven results with the original points 
        points_merge,_,_ = self.renderer.gaussians.drive_sample(points_coarse.detach())
        points_merge, _ = self.renderer.gaussians.merge(points_merge.detach(), self.renderer.gaussians.raw_xyz)

        #Get the final completed results at the end of iterations
        if itera == itere-1:
            points_coarse, grad_len, movelen = self.renderer.gaussians.drive_sample(grids)
            self.refined,_,_ = self.renderer.gaussians.drive_sample(points_coarse.detach())
            self.refined, movelen = self.renderer.gaussians.merge(self.refined, self.renderer.gaussians.raw_xyz)

        loss = 0

        #Eikonal Loss
        grad_loss = 1*(grad_len-1).square().mean()
       
        #L_{far}
        cd_far = 1*self.renderer.gaussians.chamfer(points_coarse.unsqueeze(0), self.surfpoints.unsqueeze(0))
        #L_{near}
        cd_near = 1*self.renderer.gaussians.chamfer(points_fine.unsqueeze(0), self.surfpoints.unsqueeze(0))
        #L_{mer}
        cd_merge = 1*self.renderer.gaussians.chamfer(points_merge.unsqueeze(0), self.surfpoints.unsqueeze(0)) + 0.1*self.renderer.gaussians.sigma.square().mean()
        loss = loss + 1 * cd_far +  1.0 * cd_near  + 1.0 * cd_merge + 0.0001*grad_loss 

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.synchronize()


    #Get 3D grids, far points, near points
    def rand_sample(self, points, samnum=16384, uniform=False, out=False, boxsize = 1.0):
        ptnum = points.shape[0]
        ptid = np.random.choice(ptnum,samnum,replace=True)
        points = points[ptid]

        #bounds
        minbound = points.min(0)[0].unsqueeze(0)-1e-5
        maxbound = points.max(0)[0].unsqueeze(0)+1e-5

        #Sample points near the surface points
        qsamples_near = points.detach() +  0.005 *  torch.randn(size=points.shape).cuda()

        #Get the 3D grids, far points from the surface points
        grids, qsamples_far = self.renderer.rand_sample(samnum, self.surfpoints, out, boxsize = boxsize)
        return grids.detach(), qsamples_far.detach(), qsamples_near.detach()

    def farest_sample(self,data,num):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)
        pcd = pcd.farthest_point_down_sample(num)
        return np.array(pcd.points)
            
    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

        # load prompt
        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            print(f'[INFO] load prompt from {file_prompt}...')
            with open(file_prompt, "r") as f:
                self.prompt = f.read().strip()

    def save_image(self, input_tensor, filename):
        """
        :param input_tensor: tensor
        :param filename:     """
        assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
        input_tensor = input_tensor.clone().detach()
        input_tensor = input_tensor.to(torch.device('cpu'))

        vutils.save_image(input_tensor, filename)

    #Extract surface points from the Gaussian Centers
    def extract_surface(self):
        torch.cuda.empty_cache()

        loss = 0.0
        xyz = torch.cat([self.renderer.gaussians.get_xyz, self.renderer.gaussians.raw_xyz], dim=0)
        self.countlist = 0

        #Filter points with low opacities
        noise_filter = torch.cat([self.idx.squeeze(), torch.ones(self.renderer.gaussians.raw_xyz.shape[0]).cuda()], dim=0)#.squeeze()
        nfilter = (noise_filter > 0).detach().cpu().numpy()

        for _ in range(500):
            # render random view
            ver = np.random.randint(-80, 80)
            hor = np.random.randint(-180, 180)
            radius = 0
            fov = self.fov

            pose = orbit_camera(self.angles[0]+ver, self.angles[1]+hor, 1)
            cur_cam = MiniCam(pose, self.intri[0], self.intri[1], fov, fov, 0.01, 100)
            bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")

            out = self.renderer.render(cur_cam, bg_color=bg_color,fractal=True,surf=True, nfilter=nfilter)

            self.countlist+=out['surfcount']

        #Filter points under the surface
        results = xyz[nfilter][self.countlist>0]

        #Save the surface points in the eva.ply
        write_ply_ascii_geo(os.path.join(self.zsdir,'eva.ply'), results.detach().cpu().numpy())
        return results
    
    # Completion Process
    def train(self):
        opt = self.opt
        #Reference Viewpoint Estimation
        self.pose_step()

        #Partial Gaussian Initialization
        self.renderer.defcolor(self.extri)
        self.prepare_train()
        self.color_step()
        
        #Save the colorized fixed Gaussians into ply
        path = self.zsdir + '/zscomplete_data_model.ply'
        self.renderer.gaussians.save_ply(path)

        #Processing the reference image, e.g., removing the background 
        os.system('python process.py '+self.zsdir+'/render.jpg --size 512')
    
        self.step = 0

        #self.__init__(self.opt2, self.zsdir, self.plydir, self.name)

        #Load the processed image
        self.load_input(self.zsdir+'/render_rgba.png')

        #Load the colorized Gaussian primitives
        self.renderer.init2(path=path, scale_denom = self.opt.scale_denom, z123=True)
        self.prepare_train()

        #Zero-shot-Fractal Completion
        for i in tqdm.trange(opt.zfc_num):
            self.train_step(i)
        write_ply_ascii_geo(os.path.join(self.zsdir,'gaussians.ply'), torch.cat([self.renderer.gaussians.get_xyz, self.renderer.gaussians.raw_xyz], dim=0).detach().cpu().numpy())
        
        #Gaussian Surface Extraction
        self.refined, self.idx = self.renderer.gaussians.selectpts()
        self.surfpoints = self.extract_surface().detach()
        
        #Point Cloud Extraction
        for i in tqdm.trange(opt.pce_num):
            self.refine_step(i, itere=opt.pce_num)
        self.refined = self.refined.detach().cpu().numpy() 

        #Sample the results into specific resolution and save it
        self.refined = self.farest_sample(self.refined, 16384)
        write_ply_ascii_geo(os.path.join(self.zsdir,'completed.ply'), self.refined)
        
