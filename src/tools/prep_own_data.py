import yaml
import json
import argparse
import os
import open3d as o3d
parser = argparse.ArgumentParser(
    description='Arguments for prep own data.'
)
parser.add_argument('--ouput_config', type=str, help='Path to config file.')
parser.add_argument('--scene_folder', type=str,
                    help='Path to the folder containing extracted frames and Open3d reconstruction of the scene')
args = parser.parse_args()

config = {}
config['inherit_from'] = 'configs/Own/own.yaml'

with open(os.path.join(args.scene_folder, 'intrinsic.json'), 'r') as json_file:
    intrinsic = json.load(json_file)
config['cam'] = {}
config['cam']['H'] = intrinsic['height']
config['cam']['W'] = intrinsic['width']
config['cam']['fx'] = intrinsic['intrinsic_matrix'][0]
config['cam']['fy'] = intrinsic['intrinsic_matrix'][4]
config['cam']['cx'] = intrinsic['intrinsic_matrix'][6]
config['cam']['cy'] = intrinsic['intrinsic_matrix'][7]

config['data'] = {}
config['data']['input_folder'] = args.scene_folder
config['data']['output'] = f'output/Own/{os.path.basename(args.scene_folder)}'

with open(args.ouput_config, 'w') as yaml_file:
    yaml.dump(config, yaml_file)

# since list cannot be written to the yaml the way we want it to, so we directly write strings to it
meshfile = os.path.join(args.scene_folder, 'scene', 'integrated.ply')
mesh = o3d.io.read_triangle_mesh(meshfile)
bounds = mesh.get_axis_aligned_bounding_box()
min_bound = bounds.get_min_bound()
max_bound = bounds.get_max_bound()

line = '['
for i, (mi, ma) in enumerate(zip(min_bound, max_bound)):
    line += f'[{mi-1.0},{ma+1.0}]'
    if i != 2:
        line += ','
line += ']'

with open(args.ouput_config, 'a+') as yaml_file:
    yaml_file.writelines(
        ['mapping:\n', f'\tbound: {line}\n', f'\tmarching_cubes_bound: {line}'])
