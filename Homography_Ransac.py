import os
import cv2
import random
import numpy as np
from tqdm import tqdm

from itertools import combinations
from FeatureMatching import matchfeatures


def homography(poc):
    '''
    Function to calculate homography with the given point of correspondence(poc).
    '''

    A = []
    for x, y, u, v, distance in poc:
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.asarray(A)

    # Taking SVD
    U, S, Vh = np.linalg.svd(A)

    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)

    return H


def ransac(poc, n=30, threshold=2, max_iterations=4000):
    '''
    Function to perform the ransac algorithm.

    poc: point of correspondence based on common features in the two images
    n: number of best features to apply ransac
    threshold: thresholding error for calculating inliers
    max_iterations: number of times ransac to be performed
    '''
    assert(len(poc) > n)

    best_score = 0          # To store number of inliers
    best_inliers = None     # To store inliers
    best_poc = poc[:n]      # Separating best points

    # To get list of 4 points out of best poc
    match_pairs = list(combinations(best_poc, 4))

    # Shuffling them
    random.shuffle(match_pairs)

    # Performing Ransac
    for matches in tqdm(match_pairs[:max_iterations]):

        H = homography(matches)

        inliers = []
        count = 0

        # Caclulating number of inliers
        for feature in best_poc:
            src = np.ones((3, 1))
            tgt = np.ones((3, 1))
            src[:2, 0] = feature[:2]
            tgt[:2, 0] = feature[2:4]

            # Transforming other features based on the current homography
            tgt_hat = H@src

            if tgt_hat[-1, 0] != 0:
                # Scaling to unity plane
                tgt_hat = tgt_hat/tgt_hat[-1, 0]

                # Checking if inlier
                if np.linalg.norm(tgt_hat-tgt) < threshold:
                    count += 1
                    inliers.append(feature)

        # Maintaining bucket of best inliers
        if count > best_score:
            best_score = count
            best_inliers = inliers

    # Caclulating Homography based on best inliers
    best_H = homography(best_inliers)

    print(f"Homography Results: Inliers of current H: {best_score}/{n}")

    return best_H


if __name__ == '__main__':

    IMG_DIR = r'Images_Asgnmt3_1\I1'    # Path of img directory
    N = 2                               # Number of IMages to be matched

    Images = []
    for root, dirs, files in os.walk(IMG_DIR):
        for i in range(N):
            img = cv2.imread(os.path.join(IMG_DIR, files[i]))
            img = cv2.resize(img, (800, 600))
            Images.append(img)

    # Example of the following function
    poc = matchfeatures(Images[1], Images[0])
    H = ransac(poc)

    print("The final Homography is:")
    np.set_printoptions(formatter={'float': lambda x: "{:3.2f}".format(x)})
    print(H)
