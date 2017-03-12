import glob
import cv2
import numpy as np
import matplotlib.image as mpimg

def calibrate_camera():
    images = glob.glob('./camera_cal/calibration*.jpg')

    object_points = []
    image_points = []

    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    for fname in images:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        if ret == True:
            image_points.append(corners)
            object_points.append(objp)

            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)

    img = cv2.imread('./camera_cal/calibration3.jpg')
    img_size = (img.shape[1], img.shape[0])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_size, None, None)
    print('hoooody')
    return (mtx, dist)