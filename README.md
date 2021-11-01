# MQP_MapMerging

WPI MQP 2021 - Multi-Robot Indoor Mapping

Connor Mclaughlin, Tyler Ferrara, Peter Nikopoulos

`Data/` contains the 'linear' test map from Carpin et al, "Fast and accurate map merging for multi-robot systems", resized to several resolutions. 

Example (main function of `mapmerge.py` is designed to work on 150x150 map). The output of this example can be viewed in `Examples/simple_augment.png`.

### Explanation/Motivation:
Our goal is to generate an experiment which will supply us with pairs of maps which most accurately simulate maps found in a real multi-robot exploration environment.
We generate these pairs of maps with various parameters for the amount of noise in signal (robot observations), in the form of random flips, and map rotation/translation.

These sets of maps with their augmented counterparts will serve as a benchmark dataset for map merging evaluation. 

CURRENT PIPELINE:
1. Take raw image of indoor environment from robotics dataset
2. Preprocess image into an occupancy grid with elements that can be {0: unoccupied, 1: occupied, 2: unknown, 3: outside (out of bounds)}
3. Create a copy of the occupancy grid, and mark all non-outside pixels as unknown
4. Do a simple region-filling/exploration (bfs) from a start location, to simulate robots getting calibrated in the environment
5. Simulate search in random directions from the original start region (different drones)
6. Augment (rotate+translate, include scale later) the post-search maps of each robot, and keep track of inverse augmentation
7. Attempt to merge augmented maps
8. Apply inverse augmentation on maps to check final accuracy (intersection over union) of the merge


### TODO:
1. Experiment with how robots explore their environment, i.e. let them update belief for cells in a visual cone in front of the robot, rather than just on their current cell
1b. More realistic robot movement overall (rather than random DFS). 
2. Analyze relevance of observation error and image augmentation methods (compared to real life scenario).
2. Clean Code :)