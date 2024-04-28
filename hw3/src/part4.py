import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)


def get_matches_with_orb(im1, im2):

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, desc1 = orb.detectAndCompute(im1, None)
    kp2, desc2 = orb.detectAndCompute(im2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    return points1.astype(np.int32), points2.astype(np.int32)


def transform_with_homography(H, points):

    ones = np.ones((points.shape[0], 1))
    points = np.concatenate((points, ones), axis=1)
    transformed_points = H.dot(points.T)
    transformed_points = transformed_points / (transformed_points[2, :][np.newaxis, :])
    transformed_points = transformed_points[0:2, :].T

    return transformed_points


def calculate_outlier(H, points1, points2, threshold=3):
    outliers_count = 0

    transformed_points2 = transform_with_homography(H, points2)

    x = points1[:, 0]
    y = points1[:, 1]
    x_hat = transformed_points2[:, 0]
    y_hat = transformed_points2[:, 1]
    distance = np.sqrt(np.power((x_hat - x), 2) + np.power((y_hat - y), 2)).reshape(-1)
    for dis in distance:
        if dis > threshold:
            outliers_count += 1

    return outliers_count


def ransac_for_solve_homography(points1, points2):
    # RANSAC parameters
    prob_success = 0.99
    sample_points_num = 4
    ratio_of_outlier = 0.5
    N = int(np.log(1.0 - prob_success) / np.log(1 - (1 - ratio_of_outlier) ** sample_points_num))

    # threshold for calculating outlier
    threshold = 2

    lowest_outlier_num = len(points1)  # lowest number of outliers is initialized to total points number
    best_H = None

    for _ in range(N):
        # for each iteration, random sample 4 points and solve homography
        chosen_idx = random.sample(range(len(points1)), sample_points_num)
        chosen_points1 = np.array([points1[i] for i in chosen_idx])
        chosen_points2 = np.array([points2[i] for i in chosen_idx])

        H = solve_homography(chosen_points2, chosen_points1)
        outlier_num = calculate_outlier(H, points1, points2, threshold)
        if outlier_num < lowest_outlier_num:
            lowest_outlier_num = outlier_num
            best_H = H

    return best_H


def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    canvas = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    canvas[: imgs[0].shape[0], : imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs) - 1)):
        img1 = imgs[idx]
        img2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        points1, points2 = get_matches_with_orb(img1, img2)

        # TODO: 2. apply RANSAC to choose best H
        H = ransac_for_solve_homography(points1, points2)

        # TODO: 3. chain the homographies
        last_best_H = last_best_H.dot(H)

        # TODO: 4. apply warping
        canvas = warping(img2, canvas, last_best_H, "b")

    return canvas


if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread("../resource/frame{:d}.jpg".format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite("output4.png", output4)
