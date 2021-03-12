import cv2 as cv
import cv2
import numpy as np
import os
from FeatureMatching import matchfeatures
from itertools import combinations
from tqdm import tqdm
import random


def homography(poc):

    A = []
    for x, y, u, v, distance in poc:
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H


def ransac(poc, n=30, threshold=2):
    assert(len(poc) > n)
    best_score = 0
    best_inliers = None
    best_poc = poc[:n]
    max_count = 4000
    iter_count = 0

    match_pairs = list(combinations(best_poc, 4))
    random.shuffle(match_pairs)
    for matches in tqdm(match_pairs[:max_count]):

        H = homography(matches)

        inliers = []
        count = 0

        for feature in best_poc:
            src = np.ones((3, 1))
            tgt = np.ones((3, 1))
            src[:2, 0] = feature[:2]
            tgt[:2, 0] = feature[2:4]
            tgt_hat = H@src
            if tgt_hat[-1, 0] != 0:
                tgt_hat = tgt_hat/tgt_hat[-1, 0]

                if np.linalg.norm(tgt_hat-tgt) < threshold:
                    count += 1
                    inliers.append(feature)

        if count > best_score:
            best_score = count
            best_inliers = inliers

    best_H = homography(best_inliers)
    print("Homography Results:")
    print(f"Inliers of current H: {best_score}/{n}")
    return best_H
    # print(best_H)


if __name__ == '__main__':
    IMG_DIR = r'C:\Users\jaina\Google Drive\Class\6th Sem\CV\Assignment1\PanoramaStitching\Images_Asgnmt3_1\I1'
    N = 2
    Images = []
    for root, dirs, files in os.walk(IMG_DIR):
        for i in range(N):
            img = cv2.imread(os.path.join(IMG_DIR, files[i]))
            img = cv2.resize(img, (800, 600))
            Images.append(img)

    poc = matchfeatures(Images[1], Images[0])
    H = ransac(poc)
3
