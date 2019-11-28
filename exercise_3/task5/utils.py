import os
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

def edit_scenario(scenario_path: str, y=None, d=None):
    with open(scenario_path, 'r') as infile:
        scenario = json.load(infile)

    if y is not None:
        scenario['scenario']['topography']['obstacles'][0]['shape']['y'] = y
    elif d is not None:
        targets = scenario['scenario']['topography']['targets']

        source = scenario['scenario']['topography']['sources'][0]['shape']
        mid_point = (source['y'] + source['height']) / 2

        targets[0]['shape']['y'] = mid_point + np.sqrt(d)
        targets[1]['shape']['y'] = mid_point - np.sqrt(d)

    with open(scenario_path, 'w') as outfile:
        json.dump(scenario, outfile, indent=4)


def run_simulation(vadere_root: str, scenario_path, output_path):
    os.system(f'java -jar {vadere_root}vadere-console.jar --loglevel OFF scenario-run '
              f'--scenario-file {scenario_path} --output-dir="{output_path}" --scenario-checker off')


def parse_trajectories(output_path: str, time_step_mode='fixed') -> [np.ndarray, dict]:
    """
    :param time_step_mode:
    :param output_path:
    :return: numpy array of shape (pedestrian count, 2, total time steps).
    The second dimension indicates the x and y trajectories.
    NOTE: always access pedestrian trajectory data by pedestrianId-1
    """

    trajectories = pd.read_csv(output_path, sep=' ', index_col=False)

    total_time_steps = max(trajectories['timeStep'])
    total_pedestrians = max(trajectories['pedestrianId'])

    if time_step_mode == 'fixed':
        out = np.zeros((total_pedestrians, 2, total_time_steps))
    elif time_step_mode == 'varying':
        out = {}
    else:
        raise ValueError

    for pedestrian_id in range(1, total_pedestrians+1):
        pedestrian_trajectory = trajectories[trajectories['pedestrianId'] == pedestrian_id]
        pedestrian_trajectory = pedestrian_trajectory.sort_values(by='timeStep')

        out[pedestrian_id-1] = pedestrian_trajectory[['x-PID1', 'y-PID1']].values.T

    return out


def create_plot(pedestrian_id=1):
    for y_value in os.listdir("../outputs/"):
        if y_value.startswith('.'):  # skip the hidden cache files
            continue

        trajectory_path = f'../outputs/{y_value}/{os.listdir("../outputs/"+y_value).pop()}/postvis.trajectories'
        coordinates = parse_trajectories(trajectory_path)

        xs = coordinates[pedestrian_id-1][0, :]
        ys = coordinates[pedestrian_id][1, :]
        timesteps = np.arange(0, len(xs))

        plt.figure(figsize=(15, 5))
        plt.plot(timesteps, xs, 'r', label='x')
        plt.plot(timesteps, ys, 'g', label='y')
        plt.xlabel('time steps')
        plt.ylabel('location')
        plt.legend()
        plt.title(f'Trajectory of pedestrian {pedestrian_id} for obstacle in y={y_value}')
        plt.savefig(f'../plots/task5/{pedestrian_id}_{y_value}.png', dpi=256)


def plot_phase_portrait_first_part(time_gap: int, y_values, pedestrian_id=3):
    for y_value in y_values:
        coordinates = parse_trajectories(f'../outputs/{y_value}/{os.listdir("../outputs/"+y_value).pop()}'
                                         f'/postvis.trajectories', time_step_mode='varying')
        xs = coordinates[pedestrian_id-1][0, :]

        plt.figure()
        plt.plot(xs[:-time_gap], xs[time_gap:], lw=0.3, c='blue')

        plt.title('y='+y_value)
        plt.xlabel('x at time step t')
        plt.ylabel(f'x at time step t+{time_gap}')
        plt.savefig(f'../plots/task5/{pedestrian_id}_phase_portrait_y_{y_value}.png')
        plt.close()


def plot_phase_portrait_second_part(time_gap: int, d_values, pedestrian_id=0):
    for d in d_values:
        coordinates = parse_trajectories(f'../outputs/saddle_d{d}/{os.listdir("../outputs/saddle_d"+str(d)).pop()}'
                                         f'/postvis.traj', time_step_mode='varying')
        xs = coordinates[pedestrian_id][0, :]
        ys = coordinates[pedestrian_id][1, :]

        plt.figure()
        plt.plot(xs[:-time_gap], xs[time_gap:], lw=0.3, c='blue')

        plt.title('y='+str(d))
        plt.xlabel('x at time step t')
        plt.ylabel(f'x at time step t+{time_gap}')
        plt.savefig(f'../plots/task5/{pedestrian_id}_phase_portrait_y_{d}_second_part.png')


def plot_saddle_bifurcation(d_values):
    plt.figure()
    for d in d_values:
        traj_dict: dict = parse_trajectories(f'../outputs/saddle_d{d}/{os.listdir("../outputs/saddle_d"+str(d)).pop()}'
                                             f'/postvis.traj', time_step_mode='varying')
        last_points = [traj[1, -1] for traj in traj_dict.values()]
        plt.scatter(np.full_like(last_points, fill_value=d), last_points, c='red', marker='.', s=0.2)
    plt.xlabel('d')
    plt.ylabel('y')
    plt.title('Bifurcation Diagram')
    plt.savefig('../plots/task5/saddle_bifurcation.png')

def pca_analysis():
    """
    creates a component line based on the minor component of the best case (y=4.5) using PCA
    projects every phase portrait onto that line
    plots the change of standard deviation
    """
    pedestrian_id = 12
    time_gap = 95
    std_list = []

    pca = PCA(n_components=2)
    coordinates = parse_trajectories(f'../outputs/4.5/{os.listdir("../outputs/" + str(4.5)).pop()}'
                                     f'/postvis.trajectories', time_step_mode='varying')
    xs = coordinates[pedestrian_id - 1][0, :].reshape(-1, 1)
    space = np.concatenate([xs[:-time_gap], xs[time_gap:]], axis=1)
    pca.fit(space)

    for y_value in filter(lambda f: not f.startswith('.') and not f.startswith('s'), os.listdir('../outputs')):
        coordinates = parse_trajectories(f'../outputs/{y_value}/{os.listdir("../outputs/" + str(y_value)).pop()}'
                                         f'/postvis.trajectories', time_step_mode='varying')
        xs = coordinates[pedestrian_id - 1][0, :].reshape(-1, 1)
        space = np.concatenate([xs[:-time_gap], xs[time_gap:]], axis=1)
        components = pca.transform(space)
        std_list.append((float(y_value), np.mean(components, axis=0), np.std(components, axis=0)))

    stds = list(map(lambda t: t[-1][-1], std_list))
    y_values = list(map(lambda t: float(t[0]), std_list))
    reg = LinearRegression()
    reg.fit(np.arange(3.0, 4.6, 0.1).reshape(-1, 1), stds[-16:])

    plt.figure()
    plt.plot(y_values, stds)
    plt.plot([[3.0], [4.5]], reg.predict([3.0, 4.5]), c='grey', alpha=0.5)
    plt.xticks(np.arange(2.0, 4.6, 0.1))
    plt.grid()
    plt.title('Standard Deviation of Projected Phase Portraits of Pedestrian 12')
    plt.ylabel('Standard deviation')
    plt.xlabel('y value of the obstacle')
    plt.show()