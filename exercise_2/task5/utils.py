import pandas as pd
import numpy as np
import json
import os
import shutil


def edit_scenario(scenario_path: str, target_path: str, pedestrian_data):
    """

    :param scenario_path: path of the scneario json file
    :param target_path: target path where the new scenario file should be saved
    :param pedestrian_data: a 2D numpy array where each row contains the position (and soon the velocity).
    Each row's index indicates (pedestrian's id - 1)
    """
    dynamic_elements = []

    for pedestrian_id, coordinates in enumerate(pedestrian_data, 1):
        dynamic_elements.append({
            "attributes": {
                "id": pedestrian_id,
                "radius": 0.195,
                "densityDependentSpeed": False,
                "speedDistributionMean": 1.34,
                "speedDistributionStandardDeviation": 0.26,
                "minimumSpeed": 0.5,
                "maximumSpeed": 2.2,
                "acceleration": 2.0,
                "footstepHistorySize": 4,
                "searchRadius": 1.0,
                "angleCalculationType": "USE_CENTER",
                "targetOrientationAngleThreshold": 45.0
            },
            "source": None,
            "targetIds": [1],
            "nextTargetListIndex": 0,
            "isCurrentTargetAnAgent": False,
            "position": {
                "x": coordinates[0],
                "y": coordinates[1],
            },
            "velocity": {
                "x": 0.0,
                "y": 0.0
            },
            "freeFlowSpeed": 1.420734624122518,
            "followers": [],
            "idAsTarget": -1,
            "isChild": False,
            "isLikelyInjured": False,
            "groupIds": [],
            "trajectory": {
                "footSteps": []
            },
            "groupSizes": [],
            "modelPedestrianMap": None,
            "type": "PEDESTRIAN"
        })

    with open(scenario_path, 'r') as infile:
        scenario = json.load(infile)

    scenario['scenario']['topography']['dynamicElements'] = dynamic_elements

    with open(target_path, 'w') as outfile:
        json.dump(scenario, outfile, indent=4)


def parse_trajectory(path="./bottleneck/scenarios/OSM", pedestrian_num=15, delete_output=False):
    output_dir = os.listdir(path)[0] if delete_output else ""
    pedestrian = f'_{pedestrian_num}' if not delete_output else ""
    df = pd.read_csv(f'{path}{output_dir}/postvis{pedestrian}.traj', sep=' ').apply(pd.to_numeric)

    # TODO: add 2 more values to the last axis indicating the directions of the velocity
    pedestrian_data = np.empty((max(df.timeStep), max(df.pedestrianId), 2))

    for _, row in df.iterrows():
        pedestrian_data[int(row.timeStep) - 1, int(row.pedestrianId) - 1, 0] = row["x-PID6"]
        pedestrian_data[int(row.timeStep) - 1, int(row.pedestrianId) - 1, 1] = row["y-PID6"]

    if delete_output:
        shutil.rmtree(f'{path}{output_dir}')

    return pedestrian_data


def model_prediction(true_state, scenario_path: str, target_path: str, output_path: str, vadere_root: str):
    edit_scenario(scenario_path, target_path, true_state)

    # check vadere_root and play with the quotes
    os.system(f'java -jar {vadere_root}vadere-console.jar --loglevel OFF scenario-run '
              f'--scenario-file {target_path} --output-dir="{output_path}" --scenario-checker off')

    return True
