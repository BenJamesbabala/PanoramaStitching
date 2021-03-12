import cv2 as cv
import cv2
import numpy as np
import os
from FeatureMatching import matchfeatures
from Homography_Ransac import ransac
import matplotlib.pyplot as plt
from tqdm import tqdm


def warp(src, homography, imgout, y_offset, x_offset):
    h, w, c = imgout.shape
    src_h, src_w, src_c = src.shape

    if homography is not None:
        t = homography
        homography = np.eye(3)
        for i in range(len(t)):
            homography = homography@t[i]
        pts = np.array([[0, 0, 1], [src_w, src_h, 1],
                        [src_w, 0, 1], [0, src_h, 1]]).T
        borders = (homography@pts.reshape(3, -1)).reshape(pts.shape)
        borders /= borders[-1]
        borders = (
            borders+np.array([x_offset, y_offset, 0])[:, np.newaxis]).astype(int)
        h_min, h_max = np.min(borders[1]), np.max(borders[1])
        w_min, w_max = np.min(borders[0]), np.max(borders[0])

        h_inv = np.linalg.inv(homography)
        for i in tqdm(range(h_min, h_max+1)):
            for j in range(w_min, w_max+1):
                if (0 <= i < H and 0 <= j < W):

                    u, v = i-y_offset, j-x_offset
                    src_j, src_i, scale = h_inv@np.array([v, u, 1])
                    src_i, src_j = int(src_i/scale), int(src_j/scale)

                    if(0 <= src_i < src_h and 0 <= src_j < src_w):
                        imgout[i, j] = src[src_i, src_j]

    else:
        imgout[y_offset:y_offset+src_h, x_offset:x_offset+src_w] = src

    mask = np.sum(imgout, axis=2).astype(bool)
    return imgout, mask


def laplacepyramids(images, masks, n=4):
    assert(images[0].shape[0] % n == 0 and images[0].shape[1] % n == 0)
    g_pyramids = {}
    l_pyramids = {}
    for i in range(len(images)):

        G = images[i].copy()
        g_pyramids[i] = [G]
        for _ in range(4):
            G = cv2.pyrDown(G)
            g_pyramids[i].append(G)

        l_pyramids[i] = [G]
        for j in range(len(g_pyramids[i])-2, 0, -1):
            G_up = cv2.pyrUp(G)
            G = g_pyramids[i][j]
            L = cv2.subtract(G, G_up)
            l_pyramids[i].append(L)

    print("hello")
    return l_pyramids


if __name__ == '__main__':
    IMG_DIR = r'C:\Users\jaina\Google Drive\Class\6th Sem\CV\Assignment1\PanoramaStitching\Images_Asgnmt3_1\I4'
    N = 5
    Images = []
    for root, dirs, files in os.walk(IMG_DIR):
        for i in range(N):
            img = cv2.imread(os.path.join(IMG_DIR, files[i]))
            img = cv2.resize(img, (400, 300))
            Images.append(img)

    H, W, C = 6000, 10000, 3
    img_f = np.zeros((H, W, C))
    img_outputs = []
    masks = []

    img, mask = warp(Images[N//2], None, img_f, H//2, W//2)

    img_outputs.append(img)
    masks.append(mask)
    left_H = []
    right_H = []

    for i in range(1, len(Images)//2+1):
        # right
        poc = matchfeatures(Images[N//2+i], Images[N//2+(i-1)])
        right_H.append(ransac(poc))
        img, mask = warp(Images[N//2+i], right_H, img_f, H//2, W//2)
        img_outputs.append(img)
        masks.append(mask)

        poc = matchfeatures(Images[N//2-i], Images[N//2-(i-1)])
        left_H.append(ransac(poc))
        img, mask = warp(Images[N//2-i], left_H, img_f, H//2, W//2)
        img_outputs.append(img)
        masks.append(mask)

    laplacepyramids(img_outputs, masks)

    plt.imshow(img_outputs[-1][:, :, ::-1].astype(np.uint8))
    plt.show(block=False)
    input()
