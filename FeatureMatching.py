import os
import cv2
import numpy as np


def matchfeatures(src, tgt, nfeatures=1000, verbose=False):
    '''
    src: source image
    tgt: target image
    nfeatures: number of max features for detector
    verbose: wheter to show matched features between src and tgt images

    This function matches features in the src and tgt picture to return points of 
    correspondonces of the same features between the two images

    output: [features_detected,5]
            a feature is: (x1,y1,x2,y2,distance)
    '''

    # We use ORB as our feature detector
    orb = cv2.ORB_create(nfeatures=nfeatures, scoreType=cv2.ORB_FAST_SCORE)
    kp1, des1 = orb.detectAndCompute(src, None)
    kp2, des2 = orb.detectAndCompute(tgt, None)

    # Using BruteForce Matcher to match detected features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Getting best features
    matches = sorted(matches, key=lambda x: x.distance)

    # Printing the matched features and diplaying them
    if verbose:
        net_img = cv2.drawMatches(src, kp1, tgt, kp2,
                                  matches[:20], None, flags=2)
        cv2.imshow("Keypoint Matching", net_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Storing the points of matched features
    common_points = []
    for match in matches:
        x1y1 = kp1[match.queryIdx].pt
        x2y2 = kp2[match.trainIdx].pt
        feature = list(map(int, list(x1y1) + list(x2y2) + [match.distance]))
        common_points.append(feature)

    print(f"Feature Matching Results: POC(s) found: {len(common_points)}")

    return np.array(common_points)


if __name__ == '__main__':

    IMG_DIR = r'Images_Asgnmt3_1\I1'    # Path of img directory
    N = 2                               # Number of IMages to be matched

    Images = []
    for root, dirs, files in os.walk(IMG_DIR):
        for i in range(N):
            img = cv2.imread(os.path.join(IMG_DIR, files[i]))
            img = cv2.resize(img, (800, 600))
            Images.append(img)

    # Eg to use the function
    poc = matchfeatures(Images[0], Images[1], verbose=True)
