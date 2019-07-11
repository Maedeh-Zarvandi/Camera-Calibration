import cv2
import numpy as np
‍‍‍‍import requests

url= "http://192.168.201.49:8080/shot.jpg"

while True:
    img_web=requests.get(url)
    img_arr=np.array(bytearray(img_web.content), dtype=np.uint8)
    img=cv2.imdecode(img_arr, -1)

    cv2.imshow("AndroidCam", img)

    if cv2.waitKey(1) == 27:
        break