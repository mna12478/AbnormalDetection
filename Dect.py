import cv2
import numpy as np
from scipy import io

cap = cv2.VideoCapture("1.avi")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] =0

detector = cv2.xfeatures2d.SIFT_create()  # create a feature detector
bo = cv2.BOWKMeansTrainer(20)


while (ret):
    ret, frame2 = cap.read()
    if ret:
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2', rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', rgb)
        prvs = next

        kp1, desc1 = detector.detectAndCompute(rgb, None)  # find feature

        try:
            bo.add(desc1)
            print (desc1.shape[0])
        except:
            print('exception occur')

dic = bo.cluster()
io.savemat('a.mat', {'matrix': dic})
cap.release()
cv2.destroyAllWindows()
