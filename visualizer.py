import argparse
import os
import time

import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

from src import config
from src.tools.viz import SLAMFrontend
from src.utils.datasets import get_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Arguments to visualize the SLAM process.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one inconfig file')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    parser.add_argument('--save_rendering',
                        action='store_true', help='save rendering video to `vis.mp4` in output folder ')
    parser.add_argument('--vis_input_frame',
                        action='store_true', help='visualize input frames')
    parser.add_argument('--no_gt_traj',
                        action='store_true', help='not visualize gt trajectory')
    args = parser.parse_args()
    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')
    scale = cfg['scale']
    output = cfg['data']['output'] if args.output is None else args.output
    if args.vis_input_frame:
        frame_reader = get_dataset(cfg, args, scale, device='cpu')
        frame_loader = DataLoader(
            frame_reader, batch_size=1, shuffle=False, num_workers=4)
    ckptsdir = f'{output}/ckpts'
    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f)
                 for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Get ckpt :', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            estimate_c2w_list = ckpt['estimate_c2w_list']
            gt_c2w_list = ckpt['gt_c2w_list']
            N = ckpt['idx']
    estimate_c2w_list[:, :3, 3] /= scale
    gt_c2w_list[:, :3, 3] /= scale
    estimate_c2w_list = estimate_c2w_list.cpu().numpy()
    gt_c2w_list = gt_c2w_list.cpu().numpy()

    frontend = SLAMFrontend(output, init_pose=estimate_c2w_list[0], cam_scale=0.3,
                            save_rendering=args.save_rendering, near=0,
                            estimate_c2w_list=estimate_c2w_list, gt_c2w_list=gt_c2w_list).start()

    for i in tqdm(range(0, N+1)):
        # show every second frame for speed up
        if args.vis_input_frame and i % 2 == 0:
            idx, gt_color, gt_depth, gt_c2w = frame_reader[i]
            depth_np = gt_depth.numpy()
            color_np = (gt_color.numpy()*255).astype(np.uint8)
            depth_np = depth_np/np.max(depth_np)*255
            depth_np = np.clip(depth_np, 0, 255).astype(np.uint8)
            depth_np = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
            color_np = np.clip(color_np, 0, 255)
            whole = np.concatenate([color_np, depth_np], axis=0)
            H, W, _ = whole.shape
            whole = cv2.resize(whole, (W//4, H//4))
            cv2.imshow(f'Input RGB-D Sequence', whole[:, :, ::-1])
            cv2.waitKey(1)
        time.sleep(0.03)
        meshfile = f'{output}/mesh/{i:05d}_mesh.ply'
        if os.path.isfile(meshfile):
            frontend.update_mesh(meshfile)
        frontend.update_pose(1, estimate_c2w_list[i], gt=False)
        if not args.no_gt_traj:
            frontend.update_pose(1, gt_c2w_list[i], gt=True)
        # the visualizer might get stucked if update every frame
        # with a long sequence (10000+ frames)
        if i % 10 == 0:
            frontend.update_cam_trajectory(i, gt=False)
            if not args.no_gt_traj:
                frontend.update_cam_trajectory(i, gt=True)

    if args.save_rendering:
        time.sleep(1)
        os.system(
            f"/usr/bin/ffmpeg -f image2 -r 30 -pattern_type glob -i '{output}/tmp_rendering/*.jpg' -y {output}/vis.mp4")
