import json
true = True
false = False
null = None
content = {
    "attributes" : {
      "id" : -1,
      "radius" : 0.195,
      "densityDependentSpeed" : false,
      "speedDistributionMean" : 1.34,
      "speedDistributionStandardDeviation" : 0.26,
      "minimumSpeed" : 0.5,
      "maximumSpeed" : 2.2,
      "acceleration" : 2.0,
      "footstepHistorySize" : 4,
      "searchRadius" : 1.0,
      "angleCalculationType" : "USE_CENTER",
      "targetOrientationAngleThreshold" : 45.0
    },
    "source": null,
    "targetIds": [1],
    "nextTargetListIndex": 0,
    "isCurrentTargetAnAgent": false,
    "position" : {
      "x": 12.3,
      "y": 1.8
    },
    "velocity": {
      "x": 0.0,
      "y": 0.0
    },
    "freeFlowSpeed" : 1.420734624122518,
    "followers" : [ ],
    "idAsTarget" : -1,
    "isChild" : false,
    "isLikelyInjured" : false,
    "mostImportantEvent" : null,
    "salientBehavior" : "TARGET_ORIENTED",
    "groupIds" : [ ],
    "trajectory" : {
      "footSteps" : [ ]
    },
    "groupSizes" : [ ],
    "modelPedestrianMap" : null,
    "type" : "PEDESTRIAN"
  }

path = "/Users/mm/Desktop/Data Engineering and Analysis/3. Semester/Lab Course/vadere/Scenarios/ModelTests/TestOSM/scenarios/"

with open(path+'rimea_06_corner.scenario', 'r') as infile:
    scenario = json.load(infile)



scenario['scenario']['topography']['dynamicElements'] = [content]

with open('newScenario.json', 'w') as outfile:
    json.dump(scenario, outfile, indent=4)

