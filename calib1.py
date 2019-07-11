import numpy as np
import cv2
import requests

url= "http://192.168.203.115:8080/shot.jpg"

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)


objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

while True:
    img_web=requests.get(url)
    img_arr=np.array(bytearray(img_web.content), dtype=np.uint8)
    img=cv2.imdecode(img_arr, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    temp=np.copy(gray)


    ret, corners = cv2.findChessboardCorners(gray, (7, 5), None)

    # If found, add object points, image points (after refining them)
    print(ret)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7, 5), corners2, ret)
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', 1000, 1000)
        cv2.imshow('img', img)

        cv2.waitKey(10)

        # camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        # print("mtx",mtx)
        # print("dist",dist)
        # print("rvecs",rvecs)

        # undistoring
       # img = cv2.imread('images.jpg')
        img=temp
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        # dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # crop the image
        # x, y, w, h = roi
        # dst = dst[y:y + h, x:x + w]
        # cv2.imwrite('calibresult.png', dst)
        cv2.imshow('calibrated',dst)


cv2.destroyAllWindows()