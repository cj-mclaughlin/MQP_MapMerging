import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

BASE_PATH = "Data/Mercer/"

OCCUPIED_FLOAT = 0
FREE_FLOAT = 1
UNKNOWN_FLOAT = 0.5

FREE = 255
OCCUPIED = 0
UNKNOWN = 127

TRAIN_FILENAMES = ["robot0_cast.txt", "robot1_cast.txt", "robot2_cast.txt", "robot3_cast.txt", "LongRun6.txt", "LongRun7.txt", "LongCorridor2.txt", "LongCorridor1.txt"]
TEST_FILENAMES = ["intel.txt", "intel1000.txt"]

def load_mercer_map(txt_path, dtype=np.uint8):
    """
    read mercer dataset map (.txt file) into numpy array as an occupancy grid

    output:
    2d occupancy grid
    s.t.
    0 - occupied
    1 - free
    0.5 - unknown
    """
    map = np.loadtxt(BASE_PATH+txt_path, dtype=dtype)
    if dtype != np.uint8:
        map[map == FREE] = FREE_FLOAT
        map[map == UNKNOWN] = UNKNOWN_FLOAT
    return map


def show_samples():
    for map_idx in range(len(TRAIN_FILENAMES)):
        intel = load_mercer_map(TRAIN_FILENAMES[map_idx])
        plt.imshow(intel, cmap="gray")
        plt.axis("off")
        plt.show()

def apply_warp(map, M, fill=UNKNOWN):
    map_warp = np.copy(map)
    x, y = map.shape[0], map.shape[1]
    map_warp = cv2.warpAffine(src=map_warp, M=M, dsize=(y, x), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return map_warp

def augment_map(map, shift_limit=0.2, rotate_limit=45, fill=UNKNOWN):
    """
    apply set of random image augmentation to map image
    return augmented map, as well as parameters for augmentation
    """
    x, y = map.shape[0], map.shape[1]
    center = y/2, x/2
    angle = np.random.uniform(low=-rotate_limit, high=rotate_limit)
    angle = 45  # hard code for consistency
    M_rotation = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
    rotated_map = apply_warp(map, M_rotation, fill=fill)
    shift_prop_x = np.random.uniform(low=-shift_limit, high=shift_limit)
    translation_x = shift_prop_x * x
    shift_prop_y = np.random.uniform(low=-shift_limit, high=shift_limit)
    translation_y = shift_prop_y * y
    M_translation = np.float32([
        [1, 0, translation_x],
        [0, 1, translation_y]
    ])
    augmented_map = apply_warp(rotated_map, M_translation, fill=fill)
    augment_dict = {"translation_x":translation_x, "translation_y":translation_y, "angle":angle}
    return augmented_map, augment_dict


"""### Hough"""

from skimage.transform import hough_line, hough_line_peaks

def hough_spectrum_calculation(image):
    h, theta, d = hough_map_transform(image)
    spectrum = np.sum(np.square(h), axis=0)
    max = np.max(spectrum)
    spectrum = spectrum / max
    return spectrum

def render_hough_spectrum(spectrum, image):
    # rendering the image is optional
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title('Input image')
    ax[0].set_axis_off()
    
    ax[1].set_title('Hough Spectrum')
    ax[1].set_xlabel('angle (theta)')
    ax[1].set_ylabel('hough spectrum')
    ax[1].plot(spectrum)
    plt.show()

def hough_map_transform(map):
    tested_angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
    # change map to format s.t. 1 where occupied, 0 otherwise
    edge_map = np.copy(map)
    edge_map[edge_map == OCCUPIED] = 1  # temp
    edge_map[edge_map == UNKNOWN] = 0
    edge_map[edge_map == FREE] = 0
    h, theta, d = hough_line(edge_map, theta=tested_angles)
    return h, theta, d

def FFT_circular_cross_correlation(HTM1, HTM2):
    # circular cross correlation done with FFT for O(n log N) time complexity
    cc = np.fft.ifft(np.fft.fft(np.flip(HTM1)) * np.fft.fft(np.flip(HTM2)))
    return cc / np.max(cc)

def cross_correlation(m1_spectrum, m2_spectrum):
    cc = np.fft.ifft(np.fft.fft(m2_spectrum) * np.conjugate(np.fft.fft(m1_spectrum)))
    return cc/ np.linalg.norm(cc)

def extract_local_maximums(signal, num):
    return np.argpartition(signal, -num)[-num:]

def axis_spectrum(axis, map):
    # edge_map = np.copy(map)
    # edge_map[edge_map == OCCUPIED] = 1  # temp
    # edge_map[edge_map == UNKNOWN] = 0
    # edge_map[edge_map == FREE] = 0
    spect = np.sum(map, axis=axis)
    return spect / np.linalg.norm(spect)

def compute_hypothesis(map1, map2, num):
    """
    produces best possible accuracy merges given two maps
    """
    x, y = map1.shape[0], map1.shape[1]
    center = y/2, x/2
    best_map = None
    best_acpt = -1
    best_params = None

    HS_M1 = hough_spectrum_calculation(map1)
    HS_M2 = hough_spectrum_calculation(map2)

    # TODO debug
    render_hough_spectrum(HS_M1, map1)
    render_hough_spectrum(HS_M2, map2)

    CC_M1_M2 = FFT_circular_cross_correlation(HS_M1, HS_M2)
    local_max = extract_local_maximums(CC_M1_M2, num)
    
    SX_M1 = axis_spectrum(0, map1)
    SY_M1 = axis_spectrum(1, map1)
    map3 = None

    for rot in local_max:
        M_rotation = cv2.getRotationMatrix2D(center=center, angle=rot, scale=1.0)
        map3 = apply_warp(map2, M_rotation, fill=UNKNOWN)  # dont cheat lol

        plt.imshow(map3, cmap="gray")
        plt.title("candidate map3 after just rotation")
        plt.show()

        SX_M3 = axis_spectrum(0, map3)
        SY_M3 = axis_spectrum(1, map3)

        CC_M1_M3_X = cross_correlation(SX_M1, SX_M3)
        CC_M1_M3_Y = cross_correlation(SY_M1, SY_M3)
        
        best_dx = extract_local_maximums(CC_M1_M3_X, 1)[0]
        best_dy = extract_local_maximums(CC_M1_M3_Y, 1)[0]
        
        M_translation = np.float32([
            [1, 0, best_dx],
            [0, 1, best_dy]
        ])
        cand_map = apply_warp(map3, M_translation, fill=UNKNOWN)
        acpt = accept(map1, cand_map)
        
        print(best_dx, best_dy, rot, acpt)
        
        if acpt > best_acpt:
            best_acpt = acpt
            best_map = cand_map
            best_params = (best_dx, best_dy, rot)
            
    return best_map, best_params

"""### SIFT Based"""

from tensorflow.python.keras.utils import data_utils
class CustomDataGen(data_utils.Sequence):
    def __init__(self, filenames, batch_size, shuffle=True):
        self.maps = [load_mercer_map(filename) for filename in filenames]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(filenames)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.maps)

    def __getitem__(self, index):
        map = self.maps[index]
        aug_map, labels = augment_map(map)
        # map_3d = np.expand_dims(map, axis=-1)
        # aug_map_3d = np.expand_dims(aug_map, axis=-1)
        # maps = np.concatenate([map_3d, aug_map_3d], axis=-1)
        return map, aug_map, labels
    
    def __len__(self):
        return self.n // self.batch_size


def blur_map(map):
    src = np.copy(map)
    blur = cv2.GaussianBlur(src,(3,3), sigmaX=1, sigmaY=1)
    return blur

def sift_mapmerge(map1, map2):
    map1, map2 = blur_map(map1), blur_map(map2)
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(map1, None)
    kp2, desc2 = sift.detectAndCompute(map2, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(desc1,desc2,k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    return M[:2]  # should return 2x3 homography matrix, (last row is 0 0 1 regardless)

def agr(map1, map2):
    agree = np.count_nonzero((map1 == OCCUPIED) & (map2 == OCCUPIED))
    agree += np.count_nonzero((map1 == FREE) & (map2 == OCCUPIED))
    return agree    

def dis(map1, map2):
    disagree = np.count_nonzero((map1 == OCCUPIED) & (map2 == FREE))
    disagree = np.count_nonzero((map1 == FREE) & (map2 == OCCUPIED))
    return disagree

def accept(map1, map2):
    a = agr(map1, map2)
    d = dis(map1, map2)
    iou = 0 if a == 0 else (a / (a+d))
    return iou


if __name__ == "__main__":
    # trying out SIFT...
    train_gen = CustomDataGen(filenames=TRAIN_FILENAMES, batch_size=1)
    map1, map2, params = train_gen[0]

    plt.imshow(map1, cmap="gray")
    plt.title("Original")
    plt.show()
    plt.imshow(map2, cmap="gray")
    plt.title("Rotated")
    plt.show()

    map_hough, shift_params = compute_hypothesis(map1, map2, 4)
    plt.imshow(map_hough, cmap="gray")
    plt.title("Recovered - Hough")
    plt.show()
    print("Acceptance index for hough:", accept(map1, map_hough))

    # print(shift_params)
    sift_M = sift_mapmerge(map1, map2)
    map_sift = apply_warp(map2, sift_M)
    plt.imshow(map_sift, cmap="gray")
    plt.title("Recovered - SIFT")
    plt.show()
    print("Acceptance index for SIFT:", accept(map1, map_sift))

    print("GT shift parameters:", params)
    print("Parameters found for hough:", shift_params)
    print("Parameters found for SIFT:", sift_M)