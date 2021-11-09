import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from time import time

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

def get_training_sample(map_filename):
    map = load_mercer_map(map_filename)
    aug_map, labels = augment_map(map)
    return map, aug_map

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

def augment_map(map, shift_limit=0.1, rotate_limit=360, fill=UNKNOWN):
    """
    apply set of random image augmentation to map image
    return augmented map, as well as parameters for augmentation
    """
    x, y = map.shape[0], map.shape[1]
    center = y/2, x/2
    angle = np.random.uniform(low=-rotate_limit, high=rotate_limit)
    # angle = 45  # hard code for consistency
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
    # plt.imshow(edge_map, cmap="gray")
    # plt.show()
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
    edge_map = np.copy(map)
    # edge_map[edge_map == OCCUPIED] = 1  # temp
    edge_map[edge_map == UNKNOWN] = FREE
    edge_map = 255 - edge_map
    # edge_map[edge_map == FREE] = 0
    # edge_map[edge_map == 1] = 255
    spect = np.sum(edge_map, axis=axis)
    return spect / np.max(spect)
    # return spect / np.linalg.norm(spect)

def hough_mapmerge(map1, map2, num=4):
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

    CC_M1_M2 = FFT_circular_cross_correlation(HS_M1, HS_M2)
    local_max = extract_local_maximums(CC_M1_M2, num)
    
    SX_M1 = axis_spectrum(0, map1)
    SY_M1 = axis_spectrum(1, map1)

    map3 = None
    for rot in local_max:
        M_rotation = cv2.getRotationMatrix2D(center=center, angle=-rot, scale=1.0)
        map3 = apply_warp(map2, M_rotation, fill=UNKNOWN)  # dont cheat lol
        SX_M3 = axis_spectrum(0, map3)
        SY_M3 = axis_spectrum(1, map3)

        CCX = scipy.signal.correlate(SX_M1, SX_M3, method="fft")
        CCY = scipy.signal.correlate(SY_M1, SY_M3, method="fft")
        
        best_dx = (CCX.shape[0] + (1 - SX_M1.shape[0]) - np.argmax(CCX))
        best_dy = (CCY.shape[0] + (1 - SY_M1.shape[0]) - np.argmax(CCY))
        
        M_translation = np.float32([
            [1, 0, -best_dx],
            [0, 1, -best_dy]
        ])
        cand_map = apply_warp(map3, M_translation, fill=UNKNOWN)
        acpt = accept(map1, cand_map)
        
        if acpt > best_acpt:
            best_acpt = acpt
            best_map = cand_map
            # best_params = (best_dx, best_dy, rot)
            
    return best_map

def blur_map(map):
    src = np.copy(map)
    blur = cv2.GaussianBlur(src,(3,3), sigmaX=1, sigmaY=1)
    return blur

def sift_mapmerge(map1, map2):
    map1, map2 = blur_map(map1), blur_map(map2)
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(map1, None)
    kp2, desc2 = sift.detectAndCompute(map2, None)
    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict(checks=50)
    try:
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(desc1,desc2,k=2)
        good_matches = []
        for m, n in matches:
            # lowes ratio
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    except:
        # failed merge
        return np.ones_like(map2) * UNKNOWN
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return apply_warp(map2, M[:2])

def orb_mapmerge(map1, map2):
    map1, map2 = blur_map(map1), blur_map(map2)
    orb = cv2.ORB_create(nfeatures=250)
    kp1, desc1 = orb.detectAndCompute(map1, None)
    kp2, desc2 = orb.detectAndCompute(map2, None)
    index_params = dict(algorithm = 6, trees = 6, key_size=12, multi_probe_level = 1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(desc1,desc2,k=2)
    good_matches = []
    try:
        for m, n in matches:
            # use slightly higher ratio for ORB
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)
    except:
        # failed merge
        return np.ones_like(map2) * UNKNOWN
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return apply_warp(map2, M[:2])

def agr(map1, map2):
    agree = np.count_nonzero((map1 == OCCUPIED) & (map2 == OCCUPIED))
    agree += np.count_nonzero((map1 == FREE) & (map2 == FREE))
    return agree    

def dis(map1, map2):
    disagree = np.count_nonzero((map1 == OCCUPIED) & (map2 == FREE))
    disagree += np.count_nonzero((map1 == FREE) & (map2 == OCCUPIED))
    return disagree

def accept(map1, map2):
    a = agr(map1, map2)
    d = dis(map1, map2)
    iou = 0 if a == 0 else (a / (a+d))
    return iou

if __name__ == "__main__":
    # trying out SIFT...
    filename1 = TRAIN_FILENAMES[0]
    filename2 = TRAIN_FILENAMES[5]
    filename3 = TRAIN_FILENAMES[6]
    filename4 = TEST_FILENAMES[0]

    maps = [filename1, filename2, filename3, filename4]

    N_ITERS = 1000
    SIFT_RESULTS = [[], [], [], []]
    SIFT_TIMES = [[], [], [], []]
    ORB_RESULTS = [[], [], [], []]
    ORB_TIMES = [[], [], [], []]
    HOUGH_RESULTS = [[], [], [], []]
    HOUGH_TIMES = [[], [], [], []]
    for m_idx in range(len(maps)):
        for i in range(N_ITERS):
            map1, map2 = get_training_sample(maps[m_idx])
            plt.imsave(f"Data/Images/map{m_idx}.png", map1, cmap="gray")
            # sift
            sift_start = time()
            sift_map = sift_mapmerge(map1, map2)
            sift_end = time()
            sift_elapsed = sift_start - sift_end
            SIFT_RESULTS[m_idx].append(accept(map1, sift_map))
            SIFT_TIMES[m_idx].append(sift_elapsed)
            # hough 
            hough_start = time()
            hough_map = hough_mapmerge(map1, map2)
            hough_end = time()
            hough_elapsed = hough_start - hough_end
            HOUGH_RESULTS[m_idx].append(accept(map1, hough_map))
            HOUGH_TIMES[m_idx].append(hough_elapsed)
            # orb 
            orb_start = time()
            orb_map = orb_mapmerge(map1, map2)
            orb_end = time()
            orb_elapsed = orb_start - orb_end
            ORB_RESULTS[m_idx].append(accept(map1, orb_map))
            ORB_TIMES[m_idx].append(orb_elapsed)
    
    SIFT_MU, SIFT_STD, SIFT_TIME = np.mean(SIFT_RESULTS, axis=1), np.std(SIFT_RESULTS, axis=1), np.mean(SIFT_TIMES, axis=1)
    print("(Average Acc, Std Acc, Average Time) per map: (SIFT)")
    print(SIFT_MU, SIFT_STD, SIFT_TIME)
    ORB_MU, ORB_STD, ORB_TIME = np.mean(ORB_RESULTS, axis=1), np.std(ORB_RESULTS, axis=1), np.mean(ORB_TIMES, axis=1)
    print("(Average Acc, Std Acc, Average Time) per map: (ORB)")
    print(ORB_MU, ORB_STD, ORB_TIME)
    HOUGH_MU, HOUGH_STD, HOUGH_TIME = np.mean(HOUGH_RESULTS, axis=1), np.std(HOUGH_RESULTS, axis=1), np.mean(HOUGH_TIMES, axis=1)
    print("(Average Acc, Std Acc, Average Time) per map: (HOUGH)")
    print(HOUGH_MU, HOUGH_STD, HOUGH_TIME)
