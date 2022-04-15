import argparse

import numpy as np
import torch
import trimesh
from tqdm import tqdm


def load_poses(path):
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        c2w = torch.from_numpy(c2w).float()
        poses.append(c2w)
    return poses


parser = argparse.ArgumentParser(
    description='Arguments to cull the mesh.'
)

parser.add_argument('--input_mesh', type=str,
                    help='path to the mesh to be culled')
parser.add_argument('--traj', type=str,  help='path to the trajectory')
parser.add_argument('--output_mesh', type=str,  help='path to the output mesh')
args = parser.parse_args()

H = 680
W = 1200
fx = 600.0
fy = 600.0
fx = 600.0
cx = 599.5
cy = 339.5
scale = 6553.5

poses = load_poses(args.traj)
n_imgs = len(poses)
mesh = trimesh.load(args.input_mesh, process=False)
pc = mesh.vertices
faces = mesh.faces

# delete mesh vertices that are not inside any camera's viewing frustum
whole_mask = np.ones(pc.shape[0]).astype(np.bool)
for i in tqdm(range(0, n_imgs, 1)):
    c2w = poses[i]
    points = pc.copy()
    points = torch.from_numpy(points).cuda()
    w2c = np.linalg.inv(c2w)
    w2c = torch.from_numpy(w2c).cuda().float()
    K = torch.from_numpy(
        np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)).cuda()
    ones = torch.ones_like(points[:, 0]).reshape(-1, 1).cuda()
    homo_points = torch.cat(
        [points, ones], dim=1).reshape(-1, 4, 1).cuda().float()
    cam_cord_homo = w2c@homo_points
    cam_cord = cam_cord_homo[:, :3]

    cam_cord[:, 0] *= -1
    uv = K.float()@cam_cord.float()
    z = uv[:, -1:]+1e-5
    uv = uv[:, :2]/z
    uv = uv.float().squeeze(-1).cpu().numpy()
    edge = 0
    mask = (0 <= -z[:, 0, 0].cpu().numpy()) & (uv[:, 0] < W -
                                               edge) & (uv[:, 0] > edge) & (uv[:, 1] < H-edge) & (uv[:, 1] > edge)
    whole_mask &= ~mask
pc = mesh.vertices
faces = mesh.faces
face_mask = whole_mask[mesh.faces].all(axis=1)
mesh.update_faces(~face_mask)
mesh.export(args.output_mesh)
