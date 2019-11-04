import json


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
