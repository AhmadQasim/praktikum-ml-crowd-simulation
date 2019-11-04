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
            "mostImportantEvent": None,
            "salientBehavior": "TARGET_ORIENTED",
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


def parse_trajectory(path="./bottleneck/output/OSM/", delete_output=False):
    output_dir = os.listdir(path)[0]
    df = pd.read_csv(f'{path}{output_dir}/postvis.traj', sep=' ').apply(pd.to_numeric)

    # TODO: add 2 more values to the last axis indicating the directions of the velocity
    pedestrian_data = np.empty((max(df.timeStep), max(df.pedestrianId), 2))

    for _, row in df.iterrows():
        pedestrian_data[row.timeStep - 1, row.pedestrianId - 1, 0] = row["x-PID6"]
        pedestrian_data[row.timeStep - 1, row.pedestrianId - 1, 1] = row["y-PID6"]

    if delete_output:
        shutil.rmtree(f'{path}{output_dir}')

    return pedestrian_data


def getModelPrediction(trueState, scenario_path: str, target_path: str):
    output_path = './bottleneck/output/GNM/'
    vadere_root = '"/Users/mm/Desktop/Data Engineering and Analysis/3. Semester/Lab Course/vadere/"'
    edit_scenario(scenario_path, target_path, trueState)

    os.system(f'java -jar {vadere_root}vadere-console.jar scenario-run ' # check vadere_root and play with the quotes
              f'--scenario-file {target_path} --output-dir="{output_path}"')


if __name__ == '__main__':
    path = './bottleneck/output/OSM/bottleneck_2019-11-04_18-22-38.229/bottleneck.json'
    targetPath = path[:-5] + '_gnm.json'
    output_path = './bottleneck/output/GNM/'

    ped_data = parse_trajectory()
    getModelPrediction(ped_data[0], path, targetPath)
    ped_predicted_data = parse_trajectory(output_path, delete_output=True)

    print(ped_predicted_data[0])
