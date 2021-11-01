import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUTSIDE = 3
UNKNOWN = 2
OCCUPIED = 1
UNOCCUPIED = 0

NEAREST = cv2.INTER_NEAREST

def threshold_img(img):
    """
    threshold image into 3 categories by pixel intensity
    assumes same convention as sample images from carpin et. al.
    >= 200 explored, unoccupied (0)
    <= 100 explored, occupied (1)
    100 < x < 200 unexplored (2)
    """
    img = np.asarray(img)  # ensure that it is in numpy format
    img = np.copy(img)
    img[img <= 100] = 1
    img[img >= 200] = 0
    img[img > 1] = 2
    return img

def augment_map(img, shift_limit=0.05, rotate_limit=30, fill=UNKNOWN):
    """
    apply set of random image augmentation to map image
    return augmented map, as well as parameters for augmentation
    """
    w, h = img.shape[0], img.shape[1]
    center = w/2, h/2
    angle = np.random.uniform(low=-rotate_limit, high=rotate_limit)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
    rotated_img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(w, h), flags=NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    translation_x = np.random.uniform(low=-shift_limit*w, high=shift_limit*w)
    translation_y = np.random.uniform(low=-shift_limit*w, high=shift_limit*w)
    M_translation = np.float32([
        [1, 0, translation_x],
        [0, 1, translation_y]
    ])
    augmented_img = cv2.warpAffine(rotated_img, M_translation, dsize=(w,h), flags=NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    augment_parameters = {"translation_x":translation_x, "translation_y":translation_y, "angle":angle}
    return augmented_img, augment_parameters

def get_neighbors(img, frontier):
    """
    given a map and a set of coordinates
    return the neighboring set of coordinates
    """
    neighbors = []
    x, y = frontier
    if y-1 > 0 and x-1 > 0 and img[y-1, x-1] == UNKNOWN:
        neighbors.append((x-1, y-1))
    if y-1 > 0 and img[y-1, x] == UNKNOWN:
        neighbors.append((x, y-1))
    if y-1 > 0 and x+1 < img.shape[1] and img[y-1, x+1] == UNKNOWN:
        neighbors.append((x+1, y-1))
    if x-1 > 0 and img[y, x-1] == UNKNOWN:
        neighbors.append((x-1, y))
    if x+1 < img.shape[1] and img[y, x+1] == UNKNOWN:
        neighbors.append((x+1, y))
    if y+1 < img.shape[0] and x-1 > 0 and img[y+1, x-1] == UNKNOWN:
        neighbors.append((x-1, y+1))
    if y+1 < img.shape[0] and img[y+1, x] == UNKNOWN:
        neighbors.append((x, y+1))
    if y+1 < img.shape[0] and x+1 < img.shape[1] and img[y+1, x+1] == UNKNOWN:
        neighbors.append((x+1, y+1))
    return neighbors

def region_growing(img, gt_img, x1, y1, error_rate=0.05, timesteps=1):
    """
    explore from a starting coordinate, setting all non OUTDOOR tiles as explored
    will misclassify unoccupied pixels as occupied (and vice-versa) with given probability
    expands borders by r=1 at each timestep

    ASSUMES IMG AND GT_IMG ARE ALIGNED (SAME ROTATION/TRANSLATION)

    returns the modified map image, and final frontiers.
    """
    explored_img = np.copy(img)
    toExplore = [(x1, y1)]
    seen = dict()
    newFrontiers = []
    print(np.unique(explored_img))
    for _ in range(timesteps):
        while len(toExplore) > 0:
            # visit node and record what we saw
            curr = toExplore.pop()
            if curr in seen.keys():
                continue
            seen[curr] = True
            gt_value = gt_img[curr[1], curr[0]]
            if np.random.uniform(0, 1) < error_rate:
                # flip on error
                gt_value = OCCUPIED if gt_value == UNOCCUPIED else UNOCCUPIED  
            if gt_value == OUTSIDE:
                # don't explore outside the building
                continue  
            explored_img[curr[1], curr[0]] = gt_value
            # append its neighbors to the new frontier list
            newFrontiers += get_neighbors(explored_img, curr)
        toExplore += (list(set(newFrontiers)))
    print(np.unique(explored_img))
    return explored_img, toExplore


def simulate_exploration(img, x, y, error_rate=0.05, timesteps=1):
    """
    simulates dfs search from initial start location, exploring n_timesteps cells
    marks unknown cells in path with corresponding value, with error rate

    returns list of maps of length n_frontiers
    """
    explored = np.copy(img)
    # TODO
    return explored

def show_maps(imgs, titles=""):
    """
    TODO show map with legend/titles
    """
    palette = np.array([[  0, 255,   0], # unoccupied = green
                    [  255,   0,   0],   # occupied = black
                    [0,   0,   0],   # unknown = black
                    [  0,   0, 255]])   # outside = blue

    rgbs = [palette[i] for i in imgs]
    rows = 1
    cols = len(imgs)
    axes=[]
    fig=plt.figure()

    for a in range(rows*cols):
        axes.append(fig.add_subplot(rows, cols, a+1) )
        subplot_title=("Subplot"+str(a))
        axes[-1].set_title(subplot_title)  
        plt.imshow(rgbs[a], cmap="gray")
    fig.tight_layout()    
    plt.show()

if __name__ == "__main__":
    """
    EXAMPLE EXPERIMENT FOR MAP MERGING
    PROCESS:
        1. Take raw image of indoor environment from robotics dataset
        2. Preprocess image into an occupancy grid with elements that can be {0: unoccupied, 1: occupied, 2: unknown, 3: outside (out of bounds)}
        3. Create a copy of the occupancy grid, and mark all non-outside pixels as unknown
        4. Do a simple region-filling/exploration (bfs) from a start location, to simulate robots getting calibrated in the environment
        5. Simulate search in random directions from the original start region (different drones)
        6. Augment (rotate+translate, include scale later) the post-search maps of each robot, and keep track of inverse augmentation
        7. Attempt to merge augmented maps
        8. Apply inverse augmentation on maps to check final accuracy (intersection over union) of the merge
    Currently is very messy, sorry guys I will clean this up after finishing the MVP example for our paper proposal :)
    """
    test_path = "Data/carpin.png"
    ENTRY_COORD = (195, 470)

    # load/show original image
    img = cv2.imread(test_path, 0)

    # theshold image into 3 values
    img_thresh = threshold_img(img)
    thresh = np.copy(img_thresh)

    # set unknowns as out of bounds (for our GT map)
    img_thresh[img_thresh == UNKNOWN] = OUTSIDE
    gt_map = np.copy(img_thresh)  # get copy of ground truth map

    # set all non-outside pixels to unknown (for prediction map)
    img_thresh[img_thresh != OUTSIDE] = UNKNOWN
    initial = np.copy(img_thresh)

    # grow an initial region (explore) around entrance coordinate
    region_grown = np.copy(initial)
    region_grown, final_frontiers = region_growing(region_grown, gt_map, x1=ENTRY_COORD[0], y1=ENTRY_COORD[1], error_rate=0, timesteps=100)

    # explore with 2 robots for simplicity
    start_locations = np.random.choice([i for i in range(len(final_frontiers))], size=2)
    start1, start2 = final_frontiers[start_locations[0]], final_frontiers[start_locations[1]]
    robot1_map = simulate_exploration(region_grown, x=start1[0], y=start1[1], error_rate=0, timesteps=10)
    robot2_map = simulate_exploration(region_grown, x=start2[0], y=start2[1], error_rate=0, timesteps=10)

    # augment one of the maps, pad rotation with OUTSIDE rather than UNKNOWN
    robot2_map_aug, augmentation_parameters = augment_map(robot2_map, fill=OUTSIDE)

    # visualize what we have done...
    show_maps([gt_map, initial, region_grown, robot1_map, robot2_map, robot2_map_aug])
