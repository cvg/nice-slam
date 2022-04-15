import argparse
import os
import numpy
import torch
import sys
import numpy as np
sys.path.append('.')
from src import config
from src.common import get_tensor_from_camera

def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    numpy.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = numpy.zeros((3, 3))
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:,
                         column], data_zerocentered[:, column])
    U, d, Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity(3))
    if(numpy.linalg.det(U) * numpy.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U*S*Vh
    trans = data.mean(1) - rot * model.mean(1)

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(
        alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error


def plot_traj(ax, stamps, traj, style, color, label):
    """
    Plot a trajectory using matplotlib. 

    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend

    """
    stamps.sort()
    interval = numpy.median([s-t for s, t in zip(stamps[1:], stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x) > 0:
            ax.plot(x, y, style, color=color, label=label)
            label = ""
            x = []
            y = []
        last = stamps[i]
    if len(x) > 0:
        ax.plot(x, y, style, color=color, label=label)


def evaluate_ate(first_list, second_list, plot="", _args=""):
    # parse command line
    parser = argparse.ArgumentParser(
        description='This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory.')
    # parser.add_argument('first_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    # parser.add_argument('second_file', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument(
        '--offset', help='time offset added to the timestamps of the second file (default: 0.0)', default=0.0)
    parser.add_argument(
        '--scale', help='scaling factor for the second trajectory (default: 1.0)', default=1.0)
    parser.add_argument(
        '--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)', default=0.02)
    parser.add_argument(
        '--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    parser.add_argument('--save_associations',
                        help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    parser.add_argument(
        '--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    parser.add_argument(
        '--verbose', help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)', action='store_true')
    args = parser.parse_args(_args)
    args.plot = plot
    # first_list = associate.read_file_list(args.first_file)
    # second_list = associate.read_file_list(args.second_file)

    matches = associate(first_list, second_list, float(
        args.offset), float(args.max_difference))
    if len(matches) < 2:
        raise ValueError(
            "Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! \
            Did you choose the correct sequence?")

    first_xyz = numpy.matrix(
        [[float(value) for value in first_list[a][0:3]] for a, b in matches]).transpose()
    second_xyz = numpy.matrix([[float(value)*float(args.scale)
                              for value in second_list[b][0:3]] for a, b in matches]).transpose()

    rot, trans, trans_error = align(second_xyz, first_xyz)

    second_xyz_aligned = rot * second_xyz + trans

    first_stamps = list(first_list.keys())
    first_stamps.sort()
    first_xyz_full = numpy.matrix(
        [[float(value) for value in first_list[b][0:3]] for b in first_stamps]).transpose()

    second_stamps = list(second_list.keys())
    second_stamps.sort()
    second_xyz_full = numpy.matrix([[float(value)*float(args.scale)
                                   for value in second_list[b][0:3]] for b in second_stamps]).transpose()
    second_xyz_full_aligned = rot * second_xyz_full + trans

    if args.verbose:
        print("compared_pose_pairs %d pairs" % (len(trans_error)))

        print("absolute_translational_error.rmse %f m" % numpy.sqrt(
            numpy.dot(trans_error, trans_error) / len(trans_error)))
        print("absolute_translational_error.mean %f m" %
              numpy.mean(trans_error))
        print("absolute_translational_error.median %f m" %
              numpy.median(trans_error))
        print("absolute_translational_error.std %f m" % numpy.std(trans_error))
        print("absolute_translational_error.min %f m" % numpy.min(trans_error))
        print("absolute_translational_error.max %f m" % numpy.max(trans_error))

    if args.save_associations:
        file = open(args.save_associations, "w")
        file.write("\n".join(["%f %f %f %f %f %f %f %f" % (a, x1, y1, z1, b, x2, y2, z2) for (
            a, b), (x1, y1, z1), (x2, y2, z2) in zip(matches, first_xyz.transpose().A, second_xyz_aligned.transpose().A)]))
        file.close()

    if args.save:
        file = open(args.save, "w")
        file.write("\n".join(["%f " % stamp+" ".join(["%f" % d for d in line])
                   for stamp, line in zip(second_stamps, second_xyz_full_aligned.transpose().A)]))
        file.close()

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pylab as pylab
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ATE = numpy.sqrt(
            numpy.dot(trans_error, trans_error) / len(trans_error))
        ax.set_title(f'len:{len(trans_error)} ATE RMSE:{ATE} {args.plot[:-3]}')
        plot_traj(ax, first_stamps, first_xyz_full.transpose().A,
                  '-', "black", "ground truth")
        plot_traj(ax, second_stamps, second_xyz_full_aligned.transpose(
        ).A, '-', "blue", "estimated")

        label = "difference"
        for (a, b), (x1, y1, z1), (x2, y2, z2) in zip(matches, first_xyz.transpose().A, second_xyz_aligned.transpose().A):
            # ax.plot([x1,x2],[y1,y2],'-',color="red",label=label)
            label = ""
        ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.savefig(args.plot, dpi=90)

    return {
        "compared_pose_pairs": (len(trans_error)),
        "absolute_translational_error.rmse": numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error)),
        "absolute_translational_error.mean": numpy.mean(trans_error),
        "absolute_translational_error.median": numpy.median(trans_error),
        "absolute_translational_error.std": numpy.std(trans_error),
        "absolute_translational_error.min": numpy.min(trans_error),
        "absolute_translational_error.max": numpy.max(trans_error),
    }


def evaluate(poses_gt, poses_est, plot):

    poses_gt = poses_gt.cpu().numpy()
    poses_est = poses_est.cpu().numpy()

    N = poses_gt.shape[0]
    poses_gt = dict([(i, poses_gt[i]) for i in range(N)])
    poses_est = dict([(i, poses_est[i]) for i in range(N)])

    results = evaluate_ate(poses_gt, poses_est, plot)
    print(results)


def convert_poses(c2w_list, N, scale, gt=True):
    poses = []
    mask = torch.ones(N+1).bool()
    for idx in range(0, N+1):
        if gt:
            # some frame have `nan` or `inf` in gt pose of ScanNet, 
            # but our system have estimated camera pose for all frames
            # therefore, when calculating the pose error, we need to mask out invalid pose
            if torch.isinf(c2w_list[idx]).any():
                mask[idx] = 0
                continue
            if torch.isnan(c2w_list[idx]).any():
                mask[idx] = 0
                continue
        c2w_list[idx][:3, 3] /= scale
        poses.append(get_tensor_from_camera(c2w_list[idx], Tquad=True))
    poses = torch.stack(poses)
    return poses, mask


if __name__ == '__main__':
    """
    This ATE evaluation code is modified upon the evaluation code in lie-torch.
    """

    parser = argparse.ArgumentParser(
        description='Arguments to eval the tracking ATE.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one inconfig file')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)

    args = parser.parse_args()
    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')
    scale = cfg['scale']
    output = cfg['data']['output'] if args.output is None else args.output
    cofusion = ('cofusion' in args.config) or ('CoFusion' in args.config)
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
            if cofusion:
                poses_gt = np.loadtxt(
                    'Datasets/CoFusion/room4/trajectories/gt-cam-0.txt')
                poses_gt = torch.from_numpy(poses_gt[:, 1:])
            else:
                poses_gt, mask = convert_poses(gt_c2w_list, N, scale)
            poses_est, _ = convert_poses(estimate_c2w_list, N, scale)
            poses_est = poses_est[mask]
            evaluate(poses_gt, poses_est,
                     plot=f'{output}/eval_ate_plot.png')
