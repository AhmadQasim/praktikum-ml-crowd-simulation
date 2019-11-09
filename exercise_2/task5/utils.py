import pandas as pd
import numpy as np
import json
import os
import shutil


def edit_scenario(scenario_path: str, target_path: str, pedestrian_data):
    """

    :param scenario_path: path of the scenario json file
    :param target_path: target path where the new scenario file should be saved
    :param pedestrian_data: a 2D numpy array where each row contains the position (and soon the velocity).
    Each row's index indicates (pedestrian's id - 1)
    """
    dynamic_elements = []

    for pedestrian_id, (x, y, vel_x, vel_y) in enumerate(pedestrian_data, 1):
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
                "x": x,
                "y": y,
            },
            "velocity": {
                "x": vel_x,
                "y": vel_y
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


def parse_trajectory(path, delete_output=False):
    output_dir = os.listdir(path)[1]
    position_by_timestep = pd.read_csv(f'{path}{output_dir}/postvis.traj', sep=' ', index_col=False)
    velocities = pd.read_csv(f'{path}{output_dir}/velocities.txt', sep=' ', index_col=False)

    position_by_timestep['pedestrianId'] = position_by_timestep['pedestrianId'].apply(int)

    pedestrian_data = np.empty((max(position_by_timestep.timeStep), max(position_by_timestep.pedestrianId), 4))

    for i, row in position_by_timestep.iterrows():
        pedestrian_id = row['pedestrianId']
        x = row["x-PID6"]
        y = row["y-PID6"]

        pedestrian_velocities = velocities[velocities['pedestrianId'] == pedestrian_id]
        velocity_row: pd.DataFrame = pedestrian_velocities[(pedestrian_velocities['startX-PID1'] <= x)
                                                           & (x <= pedestrian_velocities['endX-PID1'])
                                                           & (pedestrian_velocities['startY-PID1'] <= y)
                                                           & (y <= pedestrian_velocities['endY-PID1'])]

        if velocity_row.empty:
            print(f'No velocity row for x, y values of {(x, y)} and pedestrian ID of {pedestrian_id}')
            continue
        elif velocity_row.shape[0] > 1:
            print(f'Found more than 1 velocity row for x, y values of {(x, y)} and pedestrian ID of {pedestrian_id}')

        duration = velocity_row['endTime-PID1'].iloc[0] - velocity_row['simTime'].iloc[0]
        vel_x = (velocity_row['endX-PID1'].iloc[0] - velocity_row['startX-PID1'].iloc[0]) / duration
        vel_y = (velocity_row['endY-PID1'].iloc[0] - velocity_row['startY-PID1'].iloc[0]) / duration

        pedestrian_data[int(row.timeStep) - 1, int(row.pedestrianId) - 1, :] = [x, y, vel_x, vel_y]


    if delete_output:
        shutil.rmtree(f'{path}{output_dir}')

    return pedestrian_data


def getModelPrediction(trueState, scenario_path: str, target_path: str, output_path: str):
    vadere_root = '"/Users/mm/Desktop/Data Engineering and Analysis/3. Semester/Lab Course/vadere/"'
    edit_scenario(scenario_path, target_path, trueState)

    os.system(f'java -jar {vadere_root}vadere-console.jar scenario-run ' # check vadere_root and play with the quotes
              f'--scenario-file {target_path} --output-dir="{output_path}"')

def predictGNM():
    path = './bottleneck/output/OSM/model/bottleneck_2019-11-09_13-54-50.43/bottleneck.json'
    targetPath = path[:-5] + '_gnm.json'
    output_path = './bottleneck/output/GNM/prediction/'
    ped_data = parse_trajectory(path="./bottleneck/output/OSM/model/")
    getModelPrediction(ped_data[0], path, targetPath, output_path=output_path)
    ped_predicted_data = parse_trajectory(output_path, delete_output=True)
    return ped_predicted_data

def predictOSM():
    path = './bottleneck/output/GNM/model/bottleneck_2019-11-08_16-33-05.882/bottleneck.json'
    targetPath = path[:-5] + '_osm.json'
    output_path = './bottleneck/output/OSM/prediction/'
    ped_data = parse_trajectory(path="./bottleneck/output/GNM/model/")
    getModelPrediction(ped_data[0], path, targetPath, output_path=output_path)
    ped_predicted_data = parse_trajectory(output_path, delete_output=True)
    return ped_predicted_data


if __name__ == '__main__':
    # flag = 0 means OSM (ground truth) predict GNM, otherwise (GNM ground truth predict OSM) 1
    flag = 0
    if flag == 0:
        ped_predicted_data = predictGNM()
    else:
        ped_predicted_data = predictOSM()
    print(ped_predicted_data[0])
