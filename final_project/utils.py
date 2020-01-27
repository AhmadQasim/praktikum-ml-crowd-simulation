import os
import time
import torch
import numpy as np
import inspect
from contextlib import contextmanager
import subprocess
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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


def trajectory_animation(seq_data_real, seq_data, seq_start_end, sequence, path=None):
    start, end = seq_start_end[sequence]

    seq_data_real = seq_data_real[start:end]
    seq_data = seq_data[start:end]

    num_pedestrians = seq_data.shape[0]

    fig, axes = plt.subplots(1, 2, sharex='all', sharey='all')

    scatter_real = [axes[0].scatter([], []) for _ in range(num_pedestrians)]
    scatter_model = [axes[1].scatter([], []) for _ in range(num_pedestrians)]

    def animate(i):
        i = i+4
        title = "Observed" if i <= 8 else "Predicted"

        axes[0].set_title(f'Ground Truth - {title}')
        axes[1].set_title(f'GAN Model - {title}')

        for pedestrian_id in range(num_pedestrians):
            scatter_real[pedestrian_id].set_data(seq_data_real[pedestrian_id, :i, 0],
                                                 seq_data_real[pedestrian_id, :i, 1],
                                                 s=[4, 4, 4, 8])
            scatter_model[pedestrian_id].set_data(seq_data[pedestrian_id, :i, 0],
                                                  seq_data[pedestrian_id, :i, 1],
                                                  s=[4, 4, 4, 8])

    anim = FuncAnimation(fig, animate, frames=8)
    anim.save(f'{path}Sequence {sequence}.gif')










