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
    """
        For the given scenario file the dynamic elements part is overwritten by a new dynamic elements array, containing 
        every pedestrian from pedestrian data 
        with its x and y coordinate and its velocity in y and x direction
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
            "groupIds": [],
            "trajectory": {
                "footSteps": []
            },
            "groupSizes": [],
            "modelPedestrianMap": None,
            "type": "PEDESTRIAN"
        })
    # open and read given scenario file
    with open(scenario_path, 'r') as infile:
        scenario = json.load(infile)
    # overwrite the dynamic elements array of the given scenario file with the created dynamic elements array
    scenario['scenario']['topography']['dynamicElements'] = dynamic_elements

    # The created new file is saved at the given target path
    with open(target_path, 'w') as outfile:
        json.dump(scenario, outfile, indent=4)


def parse_trajectory(path, delete_output=False):
    """

    :param path: gives the path for the output files which shall be read in
     containing the output information from the different processors (timestep, pedestrian_id, x, y)
     of a simulated scenario from vadere
    :param delete_output: determines if folder in path shall be removed or not
    :return: a 2D numpy array where each row contains the position and the velocity.
    Each row's index indicates (pedestrian's id - 1)
    """
    # Read in postvis.traj file containing timestep, pedestrain_id, x,y
    position_by_timestep = pd.read_csv(f'{path}/postvis.traj', sep=' ', index_col=False)
    # Read in the velocities.txt file containing pedestrian_id, simTime, endTime, startX,endX,startY,endY
    velocities = pd.read_csv(f'{path}/velocities.txt', sep=' ', index_col=False)

    position_by_timestep['pedestrianId'] = position_by_timestep['pedestrianId'].apply(int)
    # Create numpy array with position and velocity data for each timestep and each pedestrian
    pedestrian_data = np.zeros((max(position_by_timestep.timeStep), max(position_by_timestep.pedestrianId), 4))

    for i, row in position_by_timestep.iterrows():
        # extract pedestrian and position from output file
        pedestrian_id = row['pedestrianId']
        x = row["x-PID6"]
        y = row["y-PID6"]

        # Extract the velocity data from velocities file to teh corresponding pedestrian id
        pedestrian_velocities = velocities[velocities['pedestrianId'] == pedestrian_id]
        """
        Extract only the corresponding row from the velocities file where the x coordinate and the y coordinate
        we already extracted are in between the startX and endX
        """

        velocity_row: pd.DataFrame = pedestrian_velocities[(pedestrian_velocities['startX-PID1'] <= x)
                                                           & (x <= pedestrian_velocities['endX-PID1'])
                                                           & (pedestrian_velocities['startY-PID1'] <= y)
                                                           & (y <= pedestrian_velocities['endY-PID1'])]

        # If there is no velocity data available for this x and y, save only x and y in pedestrian data
        if velocity_row.empty:
            # print(f'No velocity row for x, y values of {(x, y)} and pedestrian ID of {pedestrian_id}')
            pedestrian_data[int(row.timeStep) - 1, int(row.pedestrianId) - 1, :2] = [x, y]
            continue
        elif velocity_row.shape[0] > 1:
            pass
            # print(f'Found more than 1 velocity row for x, y values of {(x, y)} and pedestrian ID of {pedestrian_id}')

        # compute the velocity in x and y direction
        duration = velocity_row['endTime-PID1'].iloc[0] - velocity_row['simTime'].iloc[0]
        vel_x = (velocity_row['endX-PID1'].iloc[0] - velocity_row['startX-PID1'].iloc[0]) / duration
        vel_y = (velocity_row['endY-PID1'].iloc[0] - velocity_row['startY-PID1'].iloc[0]) / duration

        # set nan values to zero
        if vel_x == float('nan'):
            vel_x = 0.0
        if vel_y == float('nan'):
            vel_y = 0.0
        # save position and velocity for timestep and pedestrian
        pedestrian_data[int(row.timeStep) - 1, int(row.pedestrianId) - 1, :] = [x, y, vel_x, vel_y]

    # delete the folder in path if delete_output is true
    if delete_output:
        shutil.rmtree(path)

    return pedestrian_data


def model_prediction(true_state, scenario_path: str, target_path: str, output_path: str, vadere_root: str):
    """

    :param true_state: ground truth data about pedestrians and their position and velocity
    :param scenario_path: path of the scenario.json file
    :param target_path: path where the adjusted scenario file should be saved
    :param output_path: path of the output files from the processors
    :param vadere_root: folder of vadere jars
    :return:
    """
    edit_scenario(scenario_path, target_path, true_state)

    # check vadere_root and play with the quotes
    os.system(f'java -jar {vadere_root}vadere-console.jar --loglevel OFF scenario-run ' +
              f'--scenario-file {target_path} --output-dir="{output_path}" --scenario-checker off')


def predict(x, path_scenario, output_path, dynamic_scenario_path, vadere_root):
    """

    :param x: ground truth data from model about pedestrians and their position and velocity
    :param path_scenario: path of the scenario.json file
    :param output_path: path where the new created scenario file should be saved
    :param dynamic_scenario_path: path where the output files of the simulation should be stored
    :param vadere_root: folder of vadere jars
    :return: predicted pedestrian data (pedestrian id with position and velocity) for the next time step
    """
    model_prediction(x, path_scenario, dynamic_scenario_path, output_path=output_path, vadere_root=vadere_root)
    output_dir = os.listdir(output_path)[0]
    ped_predicted_data = parse_trajectory(os.path.join(output_path, output_dir), delete_output=True)
    return ped_predicted_data[1]
