import json


def edit_scenario(scenario_path, target_path, pedestrian_data):
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

    dynamicElements = []

    for pedestrian_id, coordinates in enumerate(pedestrian_data, 1):
        dynamicElements.append({
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
                "x": 0,
                "y": 0
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

    scenario['scenario']['topography']['dynamicElements'] = dynamicElements

    with open(target_path, 'w') as outfile:
        json.dump(scenario, outfile, indent=4)
