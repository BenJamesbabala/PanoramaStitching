import os
import cv2
import numpy as np
from tqdm import tqdm

from mywarp import warp, blend
from Homography_Ransac import ransac
from FeatureMatching import matchfeatures

if __name__ == '__main__':

    IMG_DIR = r'Images_Asgnmt3_1\I6'    # Path of img directory
    N = 5                               # Number of IMages to be matched

    Images = []
    for root, dirs, files in os.walk(IMG_DIR):
        for i in range(N):
            img = cv2.imread(os.path.join(IMG_DIR, files[i]))
            img = cv2.resize(img, (640, 480))
            Images.append(img)

    H, W, C = np.array(img.shape)*[3, N, 1]     # Finding shape of final image

    # Image Template for final image
    img_f = np.zeros((H, W, C))
    img_outputs = []
    masks = []

    print(f"\n||Setting the base image as {N//2}.||")
    img, mask = warp(Images[N//2], None, img_f.copy(), H//2, W//2)

    img_outputs.append(img)
    masks.append(mask)
    left_H = []
    right_H = []

    for i in range(1, len(Images)//2+1):

        try:
            # right
            print(f"\n||For image {N//2+i}||")
            print("Caculating POC(s)")
            poc = matchfeatures(Images[N//2+i], Images[N//2+(i-1)])
            print("Performing RANSAC")

            right_H.append(cv2.findHomography(
                poc[:, :2], poc[:, 2:4], cv2.RANSAC, 4)[0])
            print("Warping Image")
            img, mask = warp(Images[N//2+i], right_H[::-1],
                             img_f.copy(), H//2, W//2)
            img_outputs.append(img)
            masks.append(mask)
        except:
            pass

        try:
            # left
            print(f"\n||For image {N//2-i}||")
            print("Caculating POC(s)")
            poc = matchfeatures(Images[N//2-i], Images[N//2-(i-1)])
            print("Performing RANSAC")
            left_H.append(cv2.findHomography(
                poc[:, :2], poc[:, 2:4], cv2.RANSAC, 4)[0])
            print("Warping Image")
            img, mask = warp(Images[N//2-i], left_H[::-1],
                             img_f.copy(), H//2, W//2)
            img_outputs.append(img)
            masks.append(mask)
        except:
            pass

    # Blending all the images together
    print("Please wait, Image Blending...")
    uncropped = blend(img_outputs, masks)

    print("Image Blended, Final Cropping")
    # Creating a mask of the panaroma
    mask = np.sum(uncropped, axis=2).astype(bool)

    # Finding appropriate bounding box
    yy, xx = np.where(mask == 1)
    x_min, x_max = np.min(xx), np.max(xx)
    y_min, y_max = np.min(yy), np.max(yy)

    # Croping and saving
    final = uncropped[y_min:y_max, x_min:x_max]
    cv2.imwrite("Panaroma_Image.jpg", final)
    print("Succesfully Saved image as Panaroma_Image.jpg.")
