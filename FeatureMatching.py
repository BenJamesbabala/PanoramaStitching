import cv2 as cv
import cv2
import numpy as np
import os


def matchfeatures(src, tgt, nfeatures=1000):
    orb = cv2.ORB_create(nfeatures=nfeatures, scoreType=cv2.ORB_FAST_SCORE)
    kp1, des1 = orb.detectAndCompute(src, None)
    kp2, des2 = orb.detectAndCompute(tgt, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    net_img = cv2.drawMatches(src, kp1, tgt, kp2,
                              matches[:20], None, flags=2)
    # cv2.imshow("Keypoint Matching", net_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    common_pixels = []
    for match in matches:
        x1y1 = kp1[match.queryIdx].pt
        x2y2 = kp2[match.trainIdx].pt
        pixel = list(map(int, list(x1y1) + list(x2y2) + [match.distance]))
        common_pixels.append(pixel)

    return np.array(common_pixels)


if __name__ == '__main__':
    IMG_DIR = r'C:\Users\jaina\Google Drive\Class\6th Sem\CV\Assignment1\PanoramaStitching\Images_Asgnmt3_1\I1'
    N = 2
    Images = []
    for root, dirs, files in os.walk(IMG_DIR):
        for i in range(N):
            img = cv2.imread(os.path.join(IMG_DIR, files[i]))
            img = cv2.resize(img, (800, 600))
            Images.append(img)

    # for i, img in enumerate(Images):
    #     cv2.imshow(f'Image {i+1}', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    a = matchfeatures(Images[0], Images[1])

    print(hello)
