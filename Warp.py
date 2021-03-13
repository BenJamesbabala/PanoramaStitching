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
            homography = t[i]@homography
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


def laplacepyramids(images, masks, n=3):
    assert(images[0].shape[0] % n == 0 and images[0].shape[1] % n == 0)
    g_pyramids = {}
    l_pyramids = {}
    for i in range(len(images)):

        # Gaussian Pyramids
        G = images[i].copy()
        g_pyramids[i] = [G]
        for _ in range(n):
            G = cv2.pyrDown(G)
            g_pyramids[i].append(G)

        # Laplacian Pyramids
        l_pyramids[i] = [G]
        for j in range(len(g_pyramids[i])-2, -1, -1):
            G_up = cv2.pyrUp(G)
            G = g_pyramids[i][j]
            L = cv2.subtract(G, G_up)
            l_pyramids[i].append(L)

    # Blending Pyramids
    common_mask = masks[0].copy()
    common_image = images[0].copy()
    common_pyramids = [l_pyramids[0][i].copy()
                       for i in range(len(l_pyramids[0]))]

    for i in range(1, len(images)):
        mask_intersection = np.bitwise_and(common_mask, masks[i])
        if True in mask_intersection:
            y, x = np.where(mask_intersection == 1)
            x_min, x_max = np.min(x), np.max(x)
            split = ((x_max-x_min)/2 + x_min)/W

            # To decide which is left
            y1, x1 = np.where(common_mask == 1)
            y2, x2 = np.where(masks[i] == 1)

            if np.max(x1) > np.max(x2):
                left_py = l_pyramids[i]
                right_py = common_pyramids

            else:
                left_py = common_pyramids
                right_py = l_pyramids[i]

            LS = []
            for la, lb in zip(left_py, right_py):
                rows, cols, dpt = la.shape
                ls = np.hstack(
                    (la[:, 0:int(split*cols)], lb[:, int(split*cols):]))
                LS.append(ls)

        else:
            # To decide which is left
            y1, x1 = np.where(common_mask == 1)
            y2, x2 = np.where(masks[i] == 1)

            if np.max(x1) > np.max(x2):
                left_py = l_pyramids[i]
                right_py = common_pyramids

            else:
                left_py = common_pyramids
                right_py = l_pyramids[i]

            LS = []
            for la, lb in zip(left_py, right_py):
                rows, cols, dpt = la.shape
                # y, x = np.where(np.sum(np.bitwise_and(
                #     la.astype(bool), lb.astype(bool)), axis=2).astype(bool) == 1)
                # split = int((np.max(x) - np.min(x))/2 + cols/2)
                ls = la + lb
                # - \
                #     np.bitwise_and(la.astype(bool), lb.astype(bool))*(la+lb)/2
                LS.append(ls)

        ls_ = LS[0]
        for j in range(1, n+1):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.add(ls_, LS[j])

        common_image = ls_
        common_mask = np.sum(common_image.astype(bool), axis=2).astype(bool)
        common_pyramids = LS

    plt.imshow(ls_[:, :, ::-1].astype("uint8"))
    plt.show()
    return l_pyramids


if __name__ == '__main__':
    IMG_DIR = r'C:\Users\jaina\Google Drive\Class\6th Sem\CV\Assignment1\PanoramaStitching\Images_Asgnmt3_1\I5'
    N = 5
    Images = []
    for root, dirs, files in os.walk(IMG_DIR):
        for i in range(N):
            img = cv2.imread(os.path.join(IMG_DIR, files[i]))
            img = cv2.resize(img, (480, 360))
            Images.append(img)

    H, W, C = np.array(img.shape)*[3, N, 1]
    img_f = np.zeros((H, W, C))

    img_outputs = []
    masks = []

    img, mask = warp(Images[N//2], None, img_f.copy(), H//2, W//2)

    img_outputs.append(img)
    masks.append(mask)
    left_H = []
    right_H = []

    for i in range(1, len(Images)//2+1):

        try:
            # right
            poc = matchfeatures(Images[N//2+i], Images[N//2+(i-1)])
            right_H.append(ransac(poc))
            img, mask = warp(Images[N//2+i], right_H[::-1],
                             img_f.copy(), H//2, W//2)
            img_outputs.append(img)
            masks.append(mask)
        except:
            pass

        try:
            # left
            poc = matchfeatures(Images[N//2-i], Images[N//2-(i-1)])
            left_H.append(ransac(poc))
            img, mask = warp(Images[N//2-i], left_H[::-1],
                             img_f.copy(), H//2, W//2)
            img_outputs.append(img)
            masks.append(mask)
        except:
            pass

    laplacepyramids(img_outputs, masks)

    # plt.imshow(img_outputs[-1][:, :, ::-1].astype(np.uint8))
    # plt.show()
