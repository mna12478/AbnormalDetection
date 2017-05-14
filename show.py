import cv2
import numpy as np
from scipy import io
import  os


data=io.loadmat('c.mat')['matrix']
cap = cv2.VideoCapture("1.avi")

ret, frame1 = cap.read()

img1 = np.zeros((512,512,3), np.uint8)



i=0
try:
    while (ret):
        ret, frame2 = cap.read()
        if ret:
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame2', frame2)
            if(data[i][0]>0.9):#abnormal red
                img = cv2.line(img1, (i, 0), (i, 511), (0, 0, 255), 1)
            else:#normal blue
                img = cv2.line(img1, (i, 0), (i, 511), (255, 0, 0), 1)
            cv2.imshow('dect', img)
            cv2.resizeWindow('dect', 512, 10)
            i=i+1
            k = cv2.waitKey(30) & 0xff

    cap.release()
finally:
    os.system("pause")


