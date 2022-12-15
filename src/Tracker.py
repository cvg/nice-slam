import copy
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer
from liegroups.torch import SO3

class Tracker(object):
    def __init__(self, cfg, args, slam
                 ):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']

        self.idx = slam.idx
        self.nice = slam.nice
        self.bound = slam.bound
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.shared_c = slam.shared_c
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.RMI_list = slam.RMI_list
        self.RMI_cov_list = slam.RMI_cov_list
        self.tmp_RMI = slam.tmp_RMI
        self.tmp_RMI_cov = slam.tmp_RMI_cov

        self.cam_lr = cfg['tracking']['lr']
        self.device = cfg['tracking']['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.seperate_LR = cfg['tracking']['seperate_LR']
        self.w_color_loss = cfg['tracking']['w_color_loss']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.handle_dynamic = cfg['tracking']['handle_dynamic']
        self.use_color_in_tracking = cfg['tracking']['use_color_in_tracking']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']
        self.keyframe_every = cfg['mapping']['keyframe_every']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(
            self.frame_reader, batch_size=1, shuffle=False, num_workers=1)
        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def optimize_cam_in_batch(self, camera_tensor, gt_color, gt_depth, batch_size, optimizer, imu, idx):
        """
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            camera_tensor (tensor): camera tensor.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.
            imu (dict): linear/angular velocity, dt, corresponding uncertainties associated with transition from previous to current frame
            idx (int): index of current frame
        Returns:
            loss (float): The value of loss.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optimizer.zero_grad()
        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
        if self.nice:
            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(device)-det_rays_o)/det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

        ret = self.renderer.render_batch_ray(
            self.c, self.decoders, batch_rays_d, batch_rays_o,  self.device, stage='color',  gt_depth=batch_gt_depth)
        depth, uncertainty, color = ret

        uncertainty = uncertainty.detach()
        if self.handle_dynamic:
            tmp = torch.abs(batch_gt_depth-depth)/torch.sqrt(uncertainty+1e-10)
            mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0)
        else:
            mask = batch_gt_depth > 0
        if self.args.dep_u:
            loss = (torch.abs(batch_gt_depth-depth) /
                    torch.sqrt(uncertainty+1e-10 + torch.square(batch_gt_depth*0.0029)))[mask].sum()
        else:
            loss = (torch.abs(batch_gt_depth-depth) /
                    torch.sqrt(uncertainty+1e-10))[mask].sum()

        if self.use_color_in_tracking:
            color_loss = torch.abs(
                batch_gt_color - color)[mask].sum()
            loss += self.w_color_loss*color_loss

        # If using IMU, add in loss associated with current frame pose relative to prev frame pose
        # through IMU/velocity measurements
        if self.args.imu:
            # Load this in as a constant value
            # In reality, this has its own uncertainty... but let's ignore that
            c2w_prev = self.gt_c2w_list[idx-1]
            C_0 = SO3.from_matrix(c2w_prev[0:3, 0:3], normalize=True)
            r_0 = c2w_prev[0:3, 3:4]

            C_1 = SO3.from_matrix(c2w[0:3, 0:3].cpu(), normalize=True)
            r_1 = c2w[0:3, 3:4].cpu()
            
            # Compute the SO(3) difference between measured and predicted change
            gyr_tmp = (imu['gyro'] * imu['dt']).float()
            Err_C = SO3.exp( gyr_tmp ).dot( ( C_0.inv().dot(C_1) ).inv() )
            err_C = Err_C.log()

            # Compute Jacobian of error w.r.t. noise L
            L = - (torch.mm(SO3.inv_left_jacobian(err_C.detach()).double(), SO3.left_jacobian(gyr_tmp).double()) * imu['dt']).float()
            #L = - (torch.eye(3) * imu['dt']).float()
            # Form full noise covariance on gyro measurements
            Q_gyro = (imu['gyro_std']**2 * torch.eye(3)).float()
            # Compute weight on err_C
            W = torch.inverse(L @ Q_gyro @ torch.t(L))

            # Form position error by comparing velocity measurement to predicted vel meas
            err_r = (imu['vel'] - C_0.inv().dot((r_1 - r_0).squeeze() / imu['dt']).squeeze()).squeeze()

            # Overall IMU loss is sum between position and orientation loss, weighted by the
            # uncertainty on each measurement
            loss_C = (err_C.unsqueeze(0) @ W @ err_C.unsqueeze(1))[0]
            loss_r = torch.dot(err_r, err_r) / imu['vel_std']**2

            # Add up overall imu loss (my dimension handling is garbage, hence [0])
            imu_loss = loss_C[0] + loss_r[0]
            w_imu_loss = 1
            loss += w_imu_loss * imu_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.

        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')
            self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)
            for key, val in self.shared_c.items():
                val = val.clone().to(self.device)
                self.c[key] = val
            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def update_RMI(self, RMI_idx, imu, reset):
        if reset:
            self.RMI_list[RMI_idx] = torch.eye(4)
            self.RMI_cov_list[RMI_idx] = torch.zeros(6,6)
            self.tmp_RMI = []
            self.tmp_RMI_cov = []
        else:
            # Load in old RMI components
            Del_C_prev = SO3.from_matrix(self.RMI_list[RMI_idx][0:3, 0:3], normalize=True)
            Del_r_prev = self.RMI_list[RMI_idx][0:3, 3:4]

            # Precompute u * dt
            gyr_t = (imu['gyro'] * imu['dt']).float()
            vel_t = (imu['vel'] * imu['dt']).float()

            # Compute new RMI components
            Del_C_new = torch.mm(Del_C_prev.mat, SO3.exp(gyr_t).mat)
            Del_r_new = Del_r_prev + torch.mm(Del_C_prev.mat, torch.t(imu['vel'] * imu['dt']).float())
            # Save full RMI representation
            self.RMI_list[RMI_idx][0:3, 0:3] = Del_C_new
            self.RMI_list[RMI_idx][0:3, 3:4] = Del_r_new

            # Compute uncertainty propagation
            # Compute RMI Jacobian
            F_1 = torch.cat( (torch.eye(3), torch.zeros(3,3)), 1)
            
            F_2 = torch.cat( ( -SO3.wedge(torch.mm(Del_C_prev.mat, vel_t.reshape(3,1)).squeeze()), torch.eye(3)), 1)
            F_prev = torch.cat( (F_1, F_2), 0).float()

            # Compute noise Jacobian
            L_11 = (torch.mm(Del_C_prev.mat, SO3.left_jacobian(gyr_t)) * imu['dt'])
            L_22 = Del_C_prev.mat * imu['dt']
            L_prev = torch.block_diag( L_11, L_22 ).float()

            # Load in previous sigma and current IMU covariance
            Sigma_prev = self.RMI_cov_list[RMI_idx]
            Q = torch.block_diag( imu['gyro_std']**2 * torch.eye(3), imu['vel_std']**2 * torch.eye(3) ).float()

            # Propagate covariance
            self.RMI_cov_list[RMI_idx] = F_prev @ Sigma_prev @ torch.t(F_prev) + L_prev @ Q @ torch.t(L_prev)

            # Save the new RMI info up-to-idx in the list of RMIs computed since last keyframe
            # This is used in mapping in case mapping current frame is decoupled from tracking frame
            self.tmp_RMI.append(self.RMI_list[RMI_idx])
            self.tmp_RMI_cov.append(self.RMI_cov_list[RMI_idx])

    def run(self):
        device = self.device
        self.c = {}
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)

        for idx, gt_color, gt_depth, gt_c2w, imu in pbar:
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")

            idx = idx[0]
            gt_depth = gt_depth[0]
            gt_color = gt_color[0]
            gt_c2w = gt_c2w[0]

            if self.sync_method == 'strict':
                # strictly mapping and then tracking
                # initiate mapping every self.every_frame frames
                if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                    while self.mapping_idx[0] != idx-1:
                        time.sleep(0.1)
                    pre_c2w = self.estimate_c2w_list[idx-1].to(device)
            elif self.sync_method == 'loose':
                # mapping idx can be later than tracking idx is within the bound of
                # [-self.every_frame-self.every_frame//2, -self.every_frame+self.every_frame//2]
                while self.mapping_idx[0] < idx-self.every_frame-self.every_frame//2:
                    time.sleep(0.1)
            elif self.sync_method == 'free':
                # pure parallel, if mesh/vis happens may cause inbalance
                pass

            self.update_para_from_mapping()

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ", idx.item())
                print(Style.RESET_ALL)

            if idx == 0 or self.gt_camera:
                c2w = gt_c2w
                if not self.no_vis_on_first_frame:
                    self.visualizer.vis(
                        idx, 0, gt_depth, gt_color, c2w, self.c, self.decoders)

                # Initialize RMI computation to 0
                if self.args.imu:
                    # Change to 0th pose should be 0
                    self.update_RMI(0, [], reset=True)
                    # Initialize change to 1st pose
                    self.update_RMI(1, [], reset=True)
                    RMI_idx = 1

            else:
                gt_camera_tensor = get_tensor_from_camera(gt_c2w)

                # Compute RMI stuff if applicable
                if self.args.imu:
                    # Update RMIs for current index
                    self.update_RMI(RMI_idx, imu, reset=False)

                    # Check if a keyframe will be added at this idx
                    # If yes, restart RMI computation for next idx
                    if (idx % self.keyframe_every == 0 or (idx == self.n_img-2)):
                        RMI_idx += 1
                        self.update_RMI(RMI_idx, [], reset=True)

                if self.args.imu:
                    delta = torch.eye(4).to(device)
                    pre_c2w = self.estimate_c2w_list[idx-1].float().to(device)
                    gyr_t = (imu['gyro'] * imu['dt']).float()
                    vel_t = (imu['vel'] * imu['dt']).float()

                    delta[0:3, 0:3] = SO3.exp(gyr_t).mat
                    delta[0:3, 3] = vel_t

                    estimated_new_cam_c2w = pre_c2w @ delta
                else:
                    if self.const_speed_assumption and idx-2 >= 0:
                        pre_c2w = pre_c2w.float()
                        #delta = pre_c2w@self.estimate_c2w_list[idx-2].to(
                        #    device).float().inverse()
                        # Need to change this line for obelisk runtime, not sure why
                        delta = pre_c2w@self.estimate_c2w_list[idx-2].float().inverse().to(
                            device)
                        estimated_new_cam_c2w = delta@pre_c2w
                    else:
                        estimated_new_cam_c2w = pre_c2w

                camera_tensor = get_tensor_from_camera(
                    estimated_new_cam_c2w.detach())
                if self.seperate_LR:
                    camera_tensor = camera_tensor.to(device).detach()
                    T = camera_tensor[-3:]
                    quad = camera_tensor[:4]
                    cam_para_list_quad = [quad]
                    quad = Variable(quad, requires_grad=True)
                    T = Variable(T, requires_grad=True)
                    camera_tensor = torch.cat([quad, T], 0)
                    cam_para_list_T = [T]
                    cam_para_list_quad = [quad]
                    optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr},
                                                         {'params': cam_para_list_quad, 'lr': self.cam_lr*0.2}])
                else:
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    cam_para_list = [camera_tensor]
                    optimizer_camera = torch.optim.Adam(
                        cam_para_list, lr=self.cam_lr)

                initial_loss_camera_tensor = torch.abs(
                    gt_camera_tensor.to(device)-camera_tensor).mean().item()
                candidate_cam_tensor = None
                current_min_loss = 10000000000.0
                for cam_iter in range(self.num_cam_iters):
                    if self.seperate_LR:
                        camera_tensor = torch.cat([quad, T], 0).to(self.device)

                    self.visualizer.vis(
                        idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)

                    loss = self.optimize_cam_in_batch(
                        camera_tensor, gt_color, gt_depth, self.tracking_pixels, optimizer_camera, imu, idx)

                    if cam_iter == 0:
                        initial_loss = loss

                    loss_camera_tensor = torch.abs(
                        gt_camera_tensor.to(device)-camera_tensor).mean().item()
                    if self.verbose:
                        if cam_iter == self.num_cam_iters-1:
                            print(
                                f'Re-rendering loss: {initial_loss:.2f}->{loss:.2f} ' +
                                f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_tensor = camera_tensor.clone().detach()
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                    [1, 4])).type(torch.float32).to(self.device)
                c2w = get_camera_from_tensor(
                    candidate_cam_tensor.clone().detach())
                c2w = torch.cat([c2w, bottom], dim=0)
            self.estimate_c2w_list[idx] = c2w.clone().cpu()
            #print("Tracker saving c2w estimate ", idx.item())

            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()

            pre_c2w = c2w.clone()
            self.idx[0] = idx
            if self.low_gpu_mem:
                torch.cuda.empty_cache()
