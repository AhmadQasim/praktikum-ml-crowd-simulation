import pandas as pd
import numpy as np
import json
import os
import shutil


def createScenario(path, targetPath, pedestrianData):
    true = True
    false = False
    null = None

    dynamicElements = []

    for pedestrian_id, coordinates in enumerate(pedestrianData, 1):
        dynamicElements.append({
            "attributes": {
                "id": pedestrian_id,
                "radius": 0.195,
                "densityDependentSpeed": false,
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
            "source": null,
            "targetIds": [1],
            "nextTargetListIndex": 0,
            "isCurrentTargetAnAgent": false,
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
            "isChild": false,
            "isLikelyInjured": false,
            "mostImportantEvent": null,
            "salientBehavior": "TARGET_ORIENTED",
            "groupIds": [],
            "trajectory": {
                "footSteps": []
            },
            "groupSizes": [],
            "modelPedestrianMap": null,
            "type": "PEDESTRIAN"
        })

    with open(path, 'r') as infile:
        scenario = json.load(infile)

    scenario['scenario']['topography']['dynamicElements'] = dynamicElements

    with open(targetPath, 'w') as outfile:
        json.dump(scenario, outfile, indent=4)


def parseTrajectory(path="./bottleneck/output/OSM/", delete_output=False):
    output_dir = os.listdir(path)[0]
    df = pd.read_csv(path+output_dir+'/postvis.traj', sep=" ")
    pedestrian_data = np.empty((max(df.timeStep), max(df.pedestrianId), 2))
    for index, row in enumerate(df.iterrows()):
        row = row[1]
        pedestrian_data[int(row.timeStep) - 1, int(row.pedestrianId) - 1, 0] = row["x-PID6"]
        pedestrian_data[int(row.timeStep) - 1, int(row.pedestrianId) - 1, 1] = row["y-PID6"]

    if delete_output:
        shutil.rmtree(path+output_dir)
    return pedestrian_data


def editScenario(scenarioPath, targetPath, pedestrianData):
    return createScenario(scenarioPath, targetPath, pedestrianData)


def getModelPrediction(trueState, path, targetPath):
    output_path = '"./bottleneck/output/GNM/"'
    vadere_root = '"/Users/mm/Desktop/Data Engineering and Analysis/3. Semester/Lab Course/vadere/"'
    editScenario(path, targetPath, trueState)
    os.system('java -jar ' + vadere_root + 'vadere-console.jar scenario-run --scenario-file ' + targetPath
              + ' --output-dir="' + output_path + '"')


path = './bottleneck/output/OSM/bottleneck_2019-11-04_18-22-38.229/bottleneck.json'
targetPath = path + '.GNM'
output_path = './bottleneck/output/GNM/'
ped_data = parseTrajectory()
getModelPrediction(ped_data[0, :], path, targetPath)
ped_predicted_data = parseTrajectory(output_path, delete_output=True)

print(ped_predicted_data[0])
