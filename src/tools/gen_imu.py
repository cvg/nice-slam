import os
import numpy as np
import sys
import argparse
import random
import torch
from liegroups.torch import SO3

sys.path.append('/nice-slam/')
from src import config
from src.utils.datasets import get_dataset

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def gen_imu(cfg, args):
    # Arbitrarily set dt since we have no timesteps anywhere
    dt = 0.1

    # Set gravity vector in world frame
    g = torch.tensor([0.0, 0.0, -9.81])

    # Load in dataset with gt poses
    dtset = get_dataset(cfg, args, cfg['scale'])

    # Initialize measurement vectors
    gyro_meas = torch.zeros((dtset.n_img-1, 3))
    accel_meas = torch.zeros((dtset.n_img-1, 3))

    for ii in range(0, dtset.n_img-1):
        # Get current state
        c2w_ii = dtset.poses[ii]
        C_ii = SO3.from_matrix(c2w_ii[0:3, 0:3], normalize=True)
        r_ii = c2w_ii[0:3, 3:4]

        # Get next future state
        c2w_jj = dtset.poses[ii+1]
        C_jj = SO3.from_matrix(c2w_jj[0:3, 0:3], normalize=True)
        r_jj = c2w_jj[0:3, 3:4]

        if ii < dtset.n_img-2:
            # Get next next future state
            c2w_kk = dtset.poses[ii+2]
            r_kk = c2w_kk[0:3, 3:4]

            # Compute velocities
            v_ii = (r_jj - r_ii)/dt
            v_jj = (r_kk - r_jj)/dt

            # Compute accel
            a_ii = ((v_jj - v_ii)/dt).squeeze()
        else:
            # If last state, just set zero accel
            a_ii = torch.zeros((3))

        # Measured accel is C_ab.T * (a_a - g), a_ii is a_a since velocities in world frame
        accel_meas[ii,:] = C_ii.inv().dot(a_ii - g)

        # Get relative change between ii and jj body frame in R^n, divide by dt to get vel
        gyro_meas[ii,:] = (C_ii.inv().dot(C_jj)).log()/dt
        
    # Add noise
    gyro_meas = gyro_meas + np.random.multivariate_normal(np.zeros(3), args.gyro_noise*np.eye(3), dtset.n_img-1)
    accel_meas = accel_meas + np.random.multivariate_normal(np.zeros(3), args.accel_noise*np.eye(3), dtset.n_img-1)
    
    # Save dt's
    dt_list = dt * torch.ones(dtset.n_img-1)
    
    # Save all data in imu dict
    imu_data = {'gyro': gyro_meas, 'accel': accel_meas, 'dt': dt_list}

    # Save imu dict to file
    torch.save(imu_data, args.imu_dir + '/imu_data')

def main():
    setup_seed(20)

    parser = argparse.ArgumentParser(
        description='Arguments for running the IMU generation.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--accel_noise', type=float,
                        help='accel noise std value', default=0.01)
    parser.add_argument('--gyro_noise', type=float,
                        help='gyro noise std value', default=0.01)
    args = parser.parse_args()
    cfg = config.load_config(args.config, '/nice-slam/configs/nice_slam.yaml')
    args.input_folder = '/nice-slam/' + cfg['data']['input_folder']
    args.output = cfg['data']['output']

    # Make dir for imu data
    args.imu_dir = os.path.join(args.input_folder, 'imu')
    os.makedirs(args.imu_dir, exist_ok=True)

    gen_imu(cfg, args)

if __name__ == '__main__':
    main()
