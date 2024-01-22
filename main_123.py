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
import open3d as o3d
from open3d import visualization
#sys.path.append("./emd/")
#import emd_module as emd

def write_ply_ascii_geo(filedir, coords):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    f.write('end_header\n')
    #coords = coords.astype('int')
    #print(coords.shape)
    #assert False
    coords2 = np.array(coords)
    #print(coords[-1])
    #assert False
    for i in range(len(coords2)):
        p = coords[i]
        try:
            f.writelines([str(p[0]), ' ', str(p[1]), ' ',str(p[2]), '\n'])
        except:
            continue
    f.close()

    return

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

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

        self.zsdir = '/data/txhuang/data/zscomplete_data'
        self.posedir = os.path.join(self.zsdir, 'pose.npy')
        self.plydir = os.path.join(self.zsdir, 'plyobjects')
        self.name = 'cow_7'

        self.pts_path = os.path.join(self.plydir, self.name+'.ply')
        params = np.load(self.posedir, allow_pickle=True).item()[self.name]
        #print(self.name)
        #print(params.item()[self.name])
        #assert False
        self.intri = params['intrinsic']
        self.extri = params['extrinsic']
        self.angles = params['angles']
        #TR = np.array([[1,0,0,0],
        #      [0,-1,0,0],
        #      [0,0,-1,0],
        #       [0,0,0,1]]
        #      )
        #self.extri = np.dot(self.extri,TR) #
        #self.extri = np.linalg.inv(self.extri) #c2w->w2c
        
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
            #self.renderer.initialize(num_pts=self.opt.num_pts)
            #self.renderer.init2(path=self.pts_path)
            self.renderer.init2(path='/data/txhuang/data/zscomplete_data_model.ply', z123=True)
            self.points = self.renderer.gaussians._xyz.detach()
            #self.renderer.gaussians.save_ply('/root/sfs/Data/test.ply')
            #assert False

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

    def proj(self, inputs, width, height, fov, ext):
        focal = width/(2*np.tan(np.pi*90*fov))
        inputs = inputs.detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inputs)

        intrinsic = o3d.cuda.pybind.camera.PinholeCameraIntrinsic(width=width, height=height, fx=focal, fy=focal, cx=(width-1) / 2,cy=(height-1) / 2)

        TR = np.array([[1,0,0,0],
              [0,-1,0,0],
              [0,0,-1,0],
               [0,0,0,1]]
              )
        ext = np.dot(ext,TR) #
        ext = np.linalg.inv(ext) #c2w->w2c

        param = o3d.cuda.pybind.camera.PinholeCameraParameters()
        param.extrinsic = ext
        param.intrinsic = intrinsic

        vis=visualization.Visualizer()

        vis.create_window(window_name='pcd', width=width, height=height, visible=False)
        #assert False
        ctr = vis.get_view_control()
        #print(ctr)
        #assert False

        vis.add_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(param)

        opath = '/data/txhuang/data/zscomplete_data/temp.ply'
        vis.capture_depth_point_cloud(opath, do_render=True, convert_to_world_coordinate=True)
        pcd = o3d.io.read_point_cloud(opath)
        opoints = np.array(pcd.points)
        opoints = torch.tensor(opoints, dtype=torch.float32, device="cuda")

        return opoints 
    def rand_sample(self, points, samnum=16384, uniform=False):
        ptnum = points.shape[0]
        ptid = np.random.choice(ptnum,samnum,replace=True)
        points = points[ptid]

        minbound = points.min(0)[0].unsqueeze(0)-1e-5
        maxbound = points.max(0)[0].unsqueeze(0)+1e-5
        qsamples_near = points.detach() +  0.01 * (maxbound - minbound) * torch.randn(size=points.shape).cuda()
        #if uniform:
        qsamples_far = self.renderer.rand_sample(samnum, self.surfpoints)
        return qsamples_far.detach(), qsamples_near.detach()

    def prepare_train(self):

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer
        #self.sdf_optimizer = self.renderer.gaussians.sdf_optimizer

        # default camera
        #pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        #self.fixed_cam = MiniCam(
        #    pose,
        #    self.opt.ref_size,
        #    self.opt.ref_size,
        #    self.cam.fovy,
        #    self.cam.fovx,
        #    self.cam.near,
        #    self.cam.far,
        #)
        pose = orbit_camera(self.angles[0], self.angles[1], 1)
        fov = 2*np.arctan(self.intri[0]/(2*self.intri[2]))/np.pi * 180
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
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            self.guidance_zero123 = Zero123(self.device)
            print(f"[INFO] loaded zero123!")

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

    def train_step(self, itera, tune=False):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)
            #samples = self.renderer.rand_sample(self.renderer.gaussians.get_xyz.shape[0])

            loss = 0

            ### known view
            if self.input_img_torch is not None:
                cur_cam = self.fixed_cam
                out = self.renderer.render(cur_cam, fractal=True)
                #print(out["image"].shape, self.input_img_torch.shape)
                #assert False

                # rgb loss
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                #loss = loss + 10000 * step_ratio * F.mse_loss(image, self.input_img)#_torch)
                loss = loss + 10000 * step_ratio * ((image-self.input_img)).abs().mean()
                #loss = loss + 10000 * step_ratio * (out["alpha"].unsqueeze(0)*(image-self.input_img)).abs().mean()#

                # mask loss
                mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
                #loss = loss + 20000 * step_ratio * F.mse_loss(mask, self.input_mask)#_torch)
                loss = loss + 10000 * step_ratio * (mask-self.input_mask).abs().mean()

                fixed_countlist = out['surfcount']>0
                xyz = torch.cat([self.renderer.gaussians._xyz, self.renderer.gaussians.raw_xyz], dim=0)
                fixed_points = xyz[fixed_countlist]
                #write_ply_ascii_geo(os.path.join('/data/txhuang/data/zscomplete_data','temp.ply'), fixed_points.detach().cpu().numpy())
                #assert False
                fixed_cd =  100 * self.renderer.gaussians.chamfer_sum(fixed_points.unsqueeze(0), self.renderer.gaussians.raw_xyz.unsqueeze(0)) 
                loss = loss + fixed_cd
                #print(fixed_cd)

            #self.save_image(image, '/root/sfs/Data/render.jpg')
            #assert False
            #self.minbound = self.renderer.gaussians._xyz.min(0)[0].unsqueeze(0)-1e-5
            #self.naxbound = self.renderer.gaussians._xyz.max(0)[0].unsqueeze(0)-1e-5

            ### novel view (manual batch)
            render_resolution = 128 if step_ratio < 0.3 else (128 if step_ratio < 0.6 else 512)
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
            #min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
            #max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)
            min_ver = -60
            max_ver = 60
            #sdf_loss = 0
            samples_list=[]
            points_list=[]
            #self.renderer.gaussians.set_sdf(False)
            self.renderer.gaussians.sdf_network.train()
            #assert False

            for _ in range(4):

                # render random view
                #ver = np.random.randint(min_ver, max_ver)
                ver = np.random.randint(-60, 60)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                if self.enable_zero123:
                    #pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                    pose = orbit_camera(self.angles[0]+ver, self.angles[1]+hor, 1)
                    #pose = np.dot(pose, np.array(self.extri, dtype=np.float32))
                    poses.append(pose)

                    fov = 2*np.arctan(self.intri[0]/(2*self.intri[2]))/np.pi * 180
                    cur_cam = MiniCam(pose, self.intri[0], self.intri[1], fov, fov, 0.01, 100)
                    #cur_cam = MiniCam(pose, 32, 32, fov, fov, 0.01, 100)
                    #cur_cam = MiniCam(pose, render_resolution, render_resolution, fov, fov, 0.01, 100)
                    #cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                else:
                    pose = np.array(self.extri, dtype=np.float32)##data camera pose by extrinsic
                    poses.append(pose)
                    fov = 2*np.arctan(self.intri[0]/(2*self.intri[2]))/np.pi * 180
                    cur_cam = MiniCam(pose, self.intri[0], self.intri[1], fov, fov, 0.01, 100)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.renderer.render(cur_cam, bg_color=bg_color, fractal=True)

                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)
                #print(pose.shape, self.extri.shape)
                #assert False
                
                #face_points = self.proj(self.renderer.gaussians._xyz, 256, 256, fov, pose)
                #face_samples = self.rand_drive(face_points)
                #points,grad_len = self.renderer.gaussians.drive_sample(face_samples)

                #points_list.append(points.unsqueeze(0))
                #samples_list.append(face_points.unsqueeze(0))
                
                # enable mvdream training
                if self.opt.mvdream:
                    for view_i in range(1, 4):
                        pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                        poses.append(pose_i)

                        #cur_cam_i = MiniCam(pose_i, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                        cur_cam_i = MiniCam(pose_i, self.intri[0], self.intri[1], fov, fov, 0.01, 100)

                        # bg_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
                        out_i = self.renderer.render(cur_cam_i, bg_color=bg_color)

                        image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        images.append(image)

            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)
            scaling = self.renderer.gaussians.get_scaling
            #points = self.renderer.gaussians.extract_points()
            #points,grad_len = self.renderer.gaussians.drive_sample(samples)
            #samples = torch.cat(samples_list, dim=0)
            #points = torch.cat(points_list, dim=0)
            #print(samples.shape, points.shape)
            #cd_loss = 1*self.renderer.gaussians.chamfer(samples, points)
            
            self.renderer.gaussians.set_sdf(False)
            #self.renderer.gaussians.sdf_network.eval()

            #newxyz,pts_val = self.renderer.gaussians.drive_sample(self.renderer.gaussians._xyz.detach())
            #self.renderer.gaussians._xyz = 0.8*self.renderer.gaussians._xyz + 0.2*newxyz.detach()
            #if itera == 0 and tune:
            #    self.surxyz, pts_val = self.renderer.gaussians.drive_sample(self.renderer.gaussians._xyz.detach())
            #    write_ply_ascii_geo(os.path.join('/data/txhuang/data/zscomplete_data','eva.ply'), self.surxyz.detach().cpu().numpy())
                #assert False
            #self.renderer.gaussians._xyz = newxyz
            #pts_loss = pts_val.abs().mean()
            
            #cd_loss = 1*self.renderer.gaussians.chamfer(self.renderer.gaussians._xyz.unsqueeze(0), points.unsqueeze(0))
            #grad_loss = 1*(grad_len-1).square().mean()
            #fd_loss = 10000*self.renderer.gaussians.chamfer(self.renderer.gaussians.raw_xyz.unsqueeze(0), self.renderer.gaussians._xyz.unsqueeze(0))

            #print(out["visibility_filter"].sum())
            #assert False
            #scale_loss = 1000*torch.relu(scaling-0.005).mean()
            #print(scaling.shape)
            #assert False
            scale_loss = 10*scaling.mean()
            #rp_loss, nb_loss = self.renderer.gaussians.get_repulsion_loss(self.renderer.gaussians.get_xyz.unsqueeze(0))
            far_loss = self.renderer.gaussians._xyz.abs().sum(-1).max()
            opa_mean = 1*(self.renderer.gaussians.get_opacity).mean(dim=0,keepdims=True)
            opa_loss = -1000*(self.renderer.gaussians.get_opacity-opa_mean).abs().mean()+100*opa_mean.mean()
            #far_loss = ((far_loss.exp()/far_loss.exp().sum()).detach()*far_loss).sum()
            #opa_loss = 1000*(1-self.renderer.gaussians.get_opacity).abs().mean()
            #print(self.renderer.gaussians._features_dc.permute(1,0,2).shape, self.renderer.gaussians.raw_feat.permute(1,0,2).shape)
            #assert False
            col_loss = 100*self.renderer.gaussians.chamfer(self.renderer.gaussians._features_dc.permute(1,0,2), self.renderer.gaussians.raw_feat.permute(1,0,2).detach())

            #import kiui
            # print(hor, ver)
            #kiui.vis.plot_image(images)
            #print(opa_loss)

            # guidance loss
            if self.enable_sd:
                if self.opt.mvdream:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio=step_ratio if self.opt.anneal_timestep else None)
                else:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio=step_ratio if self.opt.anneal_timestep else None)

            if self.enable_zero123:
                loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio=step_ratio if self.opt.anneal_timestep else None)

            #loss += 1.0 * scale_loss #+ 10 * scaling.mean()
            loss += 1.0 * opa_loss
            loss += col_loss
            #if itera > 100:
            #loss += 1000 * far_loss
            #print(col_loss)
            #loss += 1* fd_loss
            #if itera > 300:
                #self.renderer.gaussians._xyz = 0.8*self.renderer.gaussians._xyz + 0.2*newxyz.detach()
            #    loss += 100000.0 * pts_loss#+100*rp_loss
                #print(pts_loss)
            #if tune==True:
            #    pts_loss = self.renderer.gaussians.chamfer(self.surfpoints.unsqueeze(0).detach(), self.renderer.gaussians._xyz.unsqueeze(0))
            #    loss += 100000.0 * pts_loss+1000*rp_loss
            
            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.save_image(images[:1], '/data/txhuang/data/zscomplete_data/render.jpg')

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)
        #print(scale_loss)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )

    def reset_setup(self):
        self.step = 0
        # setup training
        self.renderer.gaussians.training_setup(self.opt, refine=True)
        self.optimizer = self.renderer.gaussians.optimizer

    def refine_step(self, itera, tune=False):
        self.renderer.gaussians.set_sdf(True)
        self.renderer.gaussians.sdf_network.train()
        #starter = torch.cuda.Event(enable_timing=True)
        #ender = torch.cuda.Event(enable_timing=True)
        #starter.record()
        #uniform=False
        #if itera == 999:
        #    samnum = 16384
        #else:
        samnum = self.surfpoints.shape[0]
        samples_far, samples_near = self.rand_sample(self.surfpoints, samnum=16384, uniform=False)#.detach()
        #write_ply_ascii_geo(os.path.join('/data/txhuang/data/zscomplete_data','temp.ply'), self.renderer.gaussians.raw_xyz.detach().cpu().numpy())
        #assert False
        points_coarse, grad_len, movelen = self.renderer.gaussians.drive_sample(samples_far)
        points_fine, _, _ = self.renderer.gaussians.drive_sample(samples_near)

        #self.refined,_,_ = self.renderer.gaussians.drive_sample(points_coarse.detach())
        points_merge, movelen = self.renderer.gaussians.merge(points_fine, self.renderer.gaussians.raw_xyz)
        #colors = self.renderer.gaussians.color_mlp(points.permute(1,0).unsqueeze(0).detach())
        #colors = colors.permute(2,0,1)
        #print(points.shape, colors.shape)
        #assert False
        if itera==2999:
            self.refined,_,_ = self.renderer.gaussians.drive_sample(points_coarse.detach())
            self.refined, movelen = self.renderer.gaussians.merge(self.refined.detach(), self.renderer.gaussians.raw_xyz)
            colors, scales, opacity, rotation = self.renderer.gaussians.select_attr(self.refined, self.renderer.gaussians._xyz.detach())

        loss = 0
        ##fixed_cd = 0
        #scales = self.renderer.gaussians.neighbor_dist(self.refined)*torch.ones_like(self.renderer.gaussians.get_xyz)
        #out = self.renderer.render(self.fixed_cam, fractal=False, manual_paras = [self.refined, colors, scales, opacity, rotation], surf = False)
        #fixed_countlist = out['surfcount']>0
        #fixed_points = self.refined[fixed_countlist]
        #fixed_cd = self.renderer.gaussians.chamfer(fixed_points.unsqueeze(0), self.renderer.gaussians.raw_xyz.unsqueeze(0))

        #loss = loss + fixed_cd

        #self.newxyz = samples.clone()
        #optimizable_tensors = self.renderer.gaussians.replace_tensor_to_optimizer(self.newxyz, "xyz")
        #self.renderer.gaussians._xyz = optimizable_tensors["xyz"]
        #self.renderer.gaussians._features_dc = colors
        #assert False
        #loss = 0
        #if itera > 3000:
        #    for _ in range(self.train_steps):

        #        self.step += 1
        #        step_ratio = min(1, self.step / self.opt.iters)

        #        # update lr
        #        self.renderer.gaussians.update_learning_rate(self.step)
        #        #loss = 0

        #        ### known view
        #        if self.input_img_torch is not None:
        #            cur_cam = self.fixed_cam
        #            #print(self.refined.shape, colors.shape, scales.shape, opacity.shape, rotation.shape)
        #            #assert False
        #            out = self.renderer.render(cur_cam, fractal=False, manual_paras = [self.refined, colors, scales, opacity, rotation])
        #            #assert False

        #            # rgb loss
        #            image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
        #            #loss = loss + 10000 * step_ratio * F.mse_loss(image, self.input_img)#_torch)
        #            loss = loss + 10000 * step_ratio * ((image-self.input_img)).abs().mean()
        #            #loss = loss + 10000 * step_ratio * (out["alpha"].unsqueeze(0)*(image-self.input_img)).abs().mean()#

        #            # mask loss
        #            mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
        #            #loss = loss + 20000 * step_ratio * F.mse_loss(mask, self.input_mask)#_torch)
        #            loss = loss + 10000 * step_ratio * (mask-self.input_mask).abs().mean()

        #        ### novel view (manual batch)
        #        render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
        #        images = []
        #        poses = []
        #        vers, hors, radii = [], [], []
        #        # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
        #        min_ver = -60
        #        max_ver = 60
        #        #sdf_loss = 0
        #        samples_list=[]
        #        points_list=[]
        #        #self.renderer.gaussians.set_sdf(True)
        #        #self.renderer.gaussians.sdf_network.train()

        #        for _ in range(4):

        #            # render random view
        #            ver = np.random.randint(-60, 60)
        #            hor = np.random.randint(-180, 180)
        #            radius = 0

        #            vers.append(ver)
        #            hors.append(hor)
        #            radii.append(radius)

        #            pose = orbit_camera(self.angles[0]+ver, self.angles[1]+hor, 1)
        #            #pose = np.dot(pose, np.array(self.extri, dtype=np.float32))
        #            poses.append(pose)

        #            fov = 2*np.arctan(self.intri[0]/(2*self.intri[2]))/np.pi * 180
        #            cur_cam = MiniCam(pose, self.intri[0], self.intri[1], fov, fov, 0.01, 100)

        #            bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
        #            out = self.renderer.render(cur_cam, bg_color=bg_color, fractal=False, manual_paras = [self.refined, colors, scales, opacity, rotation])

        #            image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
        #            images.append(image)

        #    images = torch.cat(images, dim=0)
        #    loss = 0.00001*(loss + 10.0 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio=step_ratio if self.opt.anneal_timestep else None))
        #    self.save_image(images[:1], '/data/txhuang/data/zscomplete_data/render.jpg')
        #poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)
        #scaling = self.renderer.gaussians.get_scaling
        grad_loss = 1*(grad_len-1).square().mean()
        
        #scale_loss = 100*scaling.mean()
        #rp_loss, nb_loss = self.renderer.gaussians.get_repulsion_loss(self.renderer.gaussians.get_xyz.unsqueeze(0))
        ##far_loss = ((far_loss.exp()/far_loss.exp().sum()).detach()*far_loss).sum()
        #opa_loss = 1000*(1-self.renderer.gaussians.get_opacity).abs().mean()
        ##print(self.renderer.gaussians._features_dc.permute(1,0,2).shape, self.renderer.gaussians.raw_feat.permute(1,0,2).shape)
        #print(self.renderer.gaussians.get_xyz.shape,  self.surfpoints.shape)
        #assert False
        #ptnum = self.surfpoints.shape[0]
        ##kratio = 0.5
        #kid = np.random.choice(list(range(ptnum)), size=4096, replace=False)

        #if itera < 999:
        cd_far = 1*self.renderer.gaussians.chamfer(points_coarse.unsqueeze(0), self.surfpoints.unsqueeze(0))
        cd_near = 1*self.renderer.gaussians.chamfer(points_fine.unsqueeze(0), self.surfpoints.unsqueeze(0))
        cd_merge = 1*self.renderer.gaussians.chamfer(points_merge.unsqueeze(0), self.surfpoints.unsqueeze(0))+0.1*self.renderer.gaussians.sigma.square().mean()
        loss = loss + cd_far + cd_near + cd_merge + 0.001*grad_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        ##cd_loss = 10*self.renderer.gaussians.chamfer(points.unsqueeze(0), self.surfpoints.unsqueeze(0))
        #col_loss = 1*self.renderer.gaussians.chamfer(self.renderer.gaussians._features_dc.permute(1,0,2), self.renderer.gaussians.raw_feat.permute(1,0,2).detach())

        #if self.enable_zero123:
        #    loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio=step_ratio if self.opt.anneal_timestep else None)

        #loss += 1.0 * scale_loss #+ 10 * scaling.mean()
        #loss += 1.0 * opa_loss
        #loss += col_loss
        #loss = cd_loss #+ 0.1*movelen.square().mean()#+ 0.1*grad_loss
        #print(cd_loss)
        
        # optimize step
        #loss.backward()
        #self.optimizer.step()
        #self.optimizer.zero_grad()

        #self.save_image(images[:1], '/data/txhuang/data/zscomplete_data/render.jpg')

        #ender.record()
        torch.cuda.synchronize()
        #t = starter.elapsed_time(ender)

        self.need_update = True


#train sdf network
    def sdf_step(self, itera):
        #render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
        images = []
        poses = []
        vers, hors, radii = [], [], []
        # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
        #min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
        #max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)
        min_ver = -60
        max_ver = 60
        #sdf_loss = 0
        samples_list=[]
        points_list=[]
        self.renderer.gaussians.set_sdf(True)
        self.renderer.gaussians.sdf_network.train()
        loss = 0.0
        xyz = torch.cat([self.renderer.gaussians._xyz, self.renderer.gaussians.raw_xyz], dim=0)

        for _ in range(1):

            # render random view
            #ver = np.random.randint(min_ver, max_ver)
            ver = np.random.randint(-60, 60)
            hor = np.random.randint(-180, 180)
            radius = 0

            vers.append(ver)
            hors.append(hor)
            radii.append(radius)
            fov = 2*np.arctan(self.intri[0]/(2*self.intri[2]))/np.pi * 180

            pose = orbit_camera(self.angles[0]+ver, self.angles[1]+hor, 1)
            cur_cam = MiniCam(pose, self.intri[0], self.intri[1], fov, fov, 0.01, 100)
            bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
            out = self.renderer.render(cur_cam, bg_color=bg_color,fractal=True)

            face_points = xyz[out['surfcount']>0].detach()
                    #cur_cam = MiniCam(pose, 32, 32, fov, fov, 0.01, 100)
            #if itera % 10 == 0:
            #    xyz = torch.cat([self.renderer.gaussians._xyz, self.renderer.gaussians.raw_xyz], dim=0)
            #    face_points = self.proj(xyz, 256, 256, fov, pose)
            #    self.face_points = face_points
            #else:
            #    face_points = self.face_points
            #face_samples = self.rand_drive(face_points)
            #points,grad_len = self.renderer.gaussians.drive_sample(face_samples)
            #if (itera+1) % 500 == 0:
            write_ply_ascii_geo(os.path.join('/data/txhuang/data/zscomplete_data','eva.ply'), face_points.detach().cpu().numpy())
            assert False
            #write_ply_ascii_geo(os.path.join('/root/sfs/Data/zscomplete_data','xyz.ply'), face_samples.detach().cpu().numpy())

            points_list.append(points.unsqueeze(0))
            samples_list.append(face_points.unsqueeze(0))

            samples = torch.cat(samples_list, dim=0)
            points = torch.cat(points_list, dim=0)

        cd_loss = 1*self.renderer.gaussians.chamfer(samples, points)
        grad_loss = 1*(grad_len-1).square().mean()

        rp_loss = self.renderer.gaussians.get_repulsion_loss(points)

        loss += 0.5 * grad_loss
        loss += 10 * cd_loss#+10*rp_loss
        #print(grad_loss)
        #assert False
        
        # optimize step
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def extract_surface(self):
        #render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
        min_ver = -60
        max_ver = 60

        loss = 0.0
        xyz = torch.cat([self.renderer.gaussians.get_xyz, self.renderer.gaussians.raw_xyz], dim=0)
        #xyz = self.renderer.gaussians._xyz
        self.countlist = 0  
        noise_filter = torch.cat([self.idx.squeeze(), torch.ones(self.renderer.gaussians.raw_xyz.shape[0]).cuda()], dim=0)#.squeeze()

        for _ in range(500):

            # render random view
            #ver = np.random.randint(min_ver, max_ver)
            ver = np.random.randint(-60, 60)
            hor = np.random.randint(-180, 180)
            radius = 0
            fov = 2*np.arctan(self.intri[0]/(2*self.intri[2]))/np.pi * 180

            pose = orbit_camera(self.angles[0]+ver, self.angles[1]+hor, 1)
            cur_cam = MiniCam(pose, self.intri[0], self.intri[1], fov, fov, 0.01, 100)
            bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
            out = self.renderer.render(cur_cam, bg_color=bg_color,fractal=True,surf=True)

            self.countlist+=out['surfcount']
        self.countlist *= noise_filter.int()
        results = xyz[self.countlist>0]
        write_ply_ascii_geo(os.path.join('/data/txhuang/data/zscomplete_data/','eva.ply'), results.detach().cpu().numpy())
        return results

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            out = self.renderer.render(cur_cam, self.gaussain_scale_factor, fractal=True)

            buffer_image = out[self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                    self.buffer_image * (1 - self.overlay_input_img_ratio)
                    + self.input_img * self.overlay_input_img_ratio
                )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!

    
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

        # input_tensor = unnormalize(input_tensor)
        vutils.save_image(input_tensor, filename)

    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.ply')
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.' + self.opt.mesh_format)
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            #points = self.renderer.gaussians.extract_points()

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
                glctx = dr.RasterizeGLContext()
            else:
                glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                
                cur_out = self.renderer.render(cur_cam, fractal=True)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

                # enhance texture quality with zero123 [not working well]
                # if self.opt.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                    # import kiui
                    # kiui.vis.plot_image(rgbs)
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")
                
                # overlay stuff
                with dpg.group(horizontal=True):

                    def callback_toggle_overlay_input_img(sender, app_data):
                        self.overlay_input_img = not self.overlay_input_img
                        self.need_update = True

                    dpg.add_checkbox(
                        label="overlay image",
                        default_value=self.overlay_input_img,
                        callback=callback_toggle_overlay_input_img,
                    )

                    def callback_set_overlay_input_img_ratio(sender, app_data):
                        self.overlay_input_img_ratio = app_data
                        self.need_update = True

                    dpg.add_slider_float(
                        label="ratio",
                        min_value=0,
                        max_value=1,
                        format="%.1f",
                        default_value=self.overlay_input_img_ratio,
                        callback=callback_set_overlay_input_img_ratio,
                    )

                # prompt stuff
            
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )

                dpg.add_input_text(
                    label="negative",
                    default_value=self.negative_prompt,
                    callback=callback_setattr,
                    user_data="negative_prompt",
                )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        self.save_model(mode=user_data)

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_button(
                        label="geo",
                        tag="_button_save_mesh",
                        callback=callback_save,
                        user_data='geo',
                    )
                    dpg.bind_item_theme("_button_save_mesh", theme_button)

                    dpg.add_button(
                        label="geo+tex",
                        tag="_button_save_mesh_with_tex",
                        callback=callback_save,
                        user_data='geo+tex',
                    )
                    dpg.bind_item_theme("_button_save_mesh_with_tex", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    # dpg.add_button(
                    #     label="init", tag="_button_init", callback=self.prepare_train
                    # )
                    # dpg.bind_item_theme("_button_init", theme_button)

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

                def callback_set_gaussain_scale(sender, app_data):
                    self.gaussain_scale_factor = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label="gaussain scale",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                    default_value=self.gaussain_scale_factor,
                    callback=callback_set_gaussain_scale,
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            #self.extract_surface()
            #assert False
            for i in tqdm.trange(500):
                self.train_step(i)
            #self.extract_surface()
                #if (i+1) % 200 == 0:
            #for j in tqdm.trange(5000):
                #self.sdf_step(j)
            self.refined, self.idx = self.renderer.gaussians.selectpts()
            self.surfpoints = self.extract_surface().detach()

            #self.reset_setup()
            for i in tqdm.trange(3000):
                self.refine_step(i, tune=False)
                #self.train_step(i, tune=True)
            write_ply_ascii_geo(os.path.join('/data/txhuang/data/zscomplete_data','completed.ply'), self.refined.detach().cpu().numpy())
            #self.extract_surface()
            # do a last prune
            #self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        # save
        self.save_model(mode='model')
        self.save_model(mode='geo+tex')
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters)
        #gui.train(1000)
