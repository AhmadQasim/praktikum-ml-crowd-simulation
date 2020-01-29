import os
import time
import torch
import numpy as np
import inspect
from contextlib import contextmanager
import subprocess
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
The utils.py file is a slightly modified version of the original utils.py file
from GitHub
https://github.com/agrimgupta92/sgan
"""

def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def find_nan(variable, var_name):
    variable_n = variable.data.cpu().numpy()
    if np.isnan(variable_n).any():
        exit('%s has nan' % var_name)


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


def lineno():
    return str(inspect.currentframe().f_back.f_lineno)


def get_total_norm(parameters, norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm**norm_type
                total_norm = total_norm**(1. / norm_type)
            except:
                continue
    return total_norm


@contextmanager
def timeit(msg, should_time=True):
    if should_time:
        torch.cuda.synchronize()
        t0 = time.time()
    yield
    if should_time:
        torch.cuda.synchronize()
        t1 = time.time()
        duration = (t1 - t0) * 1000.0
        print('%s: %.2f ms' % (msg, duration))


def get_gpu_memory():
    torch.cuda.synchronize()
    opts = [
        'nvidia-smi', '-q', '--gpu=' + str(1), '|', 'grep', '"Used GPU Memory"'
    ]
    cmd = str.join(' ', opts)
    ps = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0].decode('utf-8')
    output = output.split("\n")[0].split(":")
    consumed_mem = int(output[1].strip().split(" ")[0])
    return consumed_mem


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    return os.path.join(_dir, 'datasets', dset_name, dset_type)


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def trajectory_animation(seq_data_real, seq_data, seq_start_end, sequence: int, prefix_path: str, args):
    start, end = seq_start_end[sequence]

    seq_data_real = seq_data_real[:, start:end].cpu().numpy()
    seq_data = seq_data[:, start:end].cpu().numpy()

    num_pedestrians = seq_data.shape[1]
    if num_pedestrians < 3:
        return

    boundary = 1

    fig, axes = plt.subplots(1, 2, sharex='all', sharey='all')

    min_x, max_x = min(seq_data_real[:, :, 0].min(), seq_data[:, :, 0].min()), max(seq_data_real[:, :, 0].max(), seq_data[:, :, 0].max())
    min_y, max_y = min(seq_data_real[:, :, 1].min(), seq_data[:, :, 1].min()), max(seq_data_real[:, :, 1].max(), seq_data[:, :, 1].max())
    axes[0].set_xlim(min_x - boundary, max_x + boundary)
    axes[0].set_ylim(min_y - boundary, max_y + boundary)
    axes[1].set_xlim(min_x - boundary, max_x + boundary)
    axes[1].set_ylim(min_y - boundary, max_y + boundary)

    scatter_real = [axes[0].plot([], []) for _ in range(num_pedestrians)]
    scatter_model = [axes[1].plot([], []) for _ in range(num_pedestrians)]

    def init():
        for pedestrian_id in range(num_pedestrians):
            sct, = scatter_real[pedestrian_id]
            sct.set_data([], [])
            sct, = scatter_model[pedestrian_id]
            sct.set_data([], [])

    def animate(i):
        i = i+4
        title = "Observed" if i < args.obs_len else "Predicted"

        axes[0].set_title(f'Ground Truth - {title}')
        axes[1].set_title(f'GAN Model - {title}')

        for pedestrian_id in range(num_pedestrians):
            sct, = scatter_real[pedestrian_id]
            sct.set_data(seq_data_real[i-4:i,pedestrian_id,  0], seq_data_real[i-4:i, pedestrian_id, 1])
            sct.set_linestyle("--" if i >= args.obs_len else "-")

            sct, = scatter_model[pedestrian_id]
            sct.set_data(seq_data[i-4:i,pedestrian_id,  0], seq_data[i-4:i, pedestrian_id, 1])
            sct.set_linestyle("--" if i >= args.obs_len else "-")


    anim = FuncAnimation(fig, animate, frames=args.obs_len+args.pred_len-4, init_func=init, interval=350)
    anim.save(f'{prefix_path}sequence_{sequence}.gif', writer='imagemagick')
    plt.close()


def animate_trajectories(loader, generator, args):
    num_samples = args.obs_len + args.pred_len
    total_traj = 0
    with torch.no_grad():
        for batch_number, batch in enumerate(loader):
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj_fake = relative_to_abs(
                pred_traj_fake_rel, obs_traj[-1]
            )

            pred_traj_full = torch.cat([obs_traj, pred_traj_fake], dim=0)
            gt_traj_full = torch.cat([obs_traj, pred_traj_gt[:args.pred_len, :, :]], dim=0)

            for seq in range(len(seq_start_end)):
                trajectory_animation(gt_traj_full,
                                     pred_traj_full,
                                     seq_start_end,
                                     seq,
                                     prefix_path=f'/home/ahmad/praktikum/praktikum_ml_crowd/final_project/animations/batch_{batch_number}',
                                     args=args)

        return ade, fde

