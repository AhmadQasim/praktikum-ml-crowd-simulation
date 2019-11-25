import os
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt


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


def parse_trajectories(output_path: str) -> np.ndarray:
    '''

    :param output_path:
    :return: numpy array of shape (pedestrian count, 2, total time steps).
    The second dimension indicates the x and y trajectories.
    NOTE: always access pedestrian trajectory data by pedestrianId-1
    '''
    trajectories = pd.read_csv(output_path, sep=' ', index_col=False)

    total_time_steps = max(trajectories['timeStep'])
    total_pedestrians = max(trajectories['pedestrianId'])

    out = np.zeros((total_pedestrians, 2, total_time_steps))

    for pedestrian_id in range(1, total_pedestrians+1):
        pedestrian_trajectory = trajectories[trajectories['pedestrianId'] == pedestrian_id]
        pedestrian_trajectory.sort_values(by='timeStep', inplace=True)

        out[pedestrian_id-1] = pedestrian_trajectory[['x-PID1', 'y-PID1']].values.T

    return out


def createPlot(pedestrian_id=1):
    for y_value in os.listdir("./outputs/"):
        if y_value.startswith('.'):  # skip the hidden cache files
            continue

        trajectory_path = f'./outputs/{y_value}/{os.listdir("./outputs/"+y_value).pop()}/postvis.trajectories'
        coordinates = parse_trajectories(trajectory_path)

        xs = coordinates[pedestrian_id-1, 0, :]
        ys = coordinates[pedestrian_id, 1, :]
        timesteps = np.arange(0, len(xs))

        plt.figure(figsize=(15, 5))
        plt.plot(timesteps, xs, 'r', label='x')
        plt.plot(timesteps, ys, 'g', label='y')
        plt.xlabel('time steps')
        plt.ylabel('location')
        plt.legend()
        plt.title(f'Trajectory of pedestrian {pedestrian_id} for obstacle in y={y_value}')
        plt.savefig(f'plots/task5/{pedestrian_id}_{y_value}.png', dpi=256)


def plot_phase_portrait(time_gap: int, y_values, pedestrian_id=3):
    for y_value in y_values:
        coordinates = parse_trajectories(f'./outputs/{y_value}/{os.listdir("./outputs/"+y_value).pop()}/postvis.trajectories')
        xs = coordinates[pedestrian_id-1, 0, :]

        plt.figure()
        plt.plot(xs[:-time_gap], xs[time_gap:], lw=0.3, c='blue')

        plt.title('y='+y_value)
        plt.xlabel('x at time step t')
        plt.ylabel(f'x at time step t+{time_gap}')
        plt.savefig(f'./plots/task5/{pedestrian_id}_phase_portrait_y_{y_value}.png')


if __name__ == '__main__':
    #plot_phase_portrait(163-68, filter(lambda f: not f.startswith('.'), os.listdir('./outputs/')), pedestrian_id=3)
    #createPlot(3)
    #y = 4.5
    #edit_scenario("Bottleneck bifurcation.scenario", y=y)
    #run_simulation("'/Users/mm/Desktop/Data Engineering and Analysis/3. Semester/Lab Course/vadere/'",
                      # "'Bottleneck bifurcation.scenario'",
                      # f'outputs/{y}/')

    for d in np.arange(0.0, 5.0, 0.5):
        edit_scenario("./Saddle_Node_Bifurcation.scenario", d=d)
        run_simulation('"C:/Users/Kaan/Desktop/vadere/"', "./Saddle_Node_Bifurcation.scenario", f'./outputs/saddle_d{d}/')
