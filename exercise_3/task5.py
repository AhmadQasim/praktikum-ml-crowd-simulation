import os
import pandas as pd
import json
import numpy as np


def edit_scenario(scenario_path: str, y=2.0):
    with open(scenario_path, 'r') as infile:
        scenario = json.load(infile)

    scenario['scenario']['topography']['obstacles'][0]['shape']['y'] = y

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

    for pedestrian_id in range(1, total_time_steps+1):
        pedestrian_trajectory = trajectories[trajectories['pedestrianId'] == pedestrian_id]
        pedestrian_trajectory.sort_values(by='timeStep', inplace=True)
        out[pedestrian_id-1] = pedestrian_trajectory[['x-PID1', 'y-PID1']].values.T

    return out


if __name__ == '__main__':
    pass
