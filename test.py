import numpy as np
import cv2

# Create a black image
img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
img = cv2.line(img,(0,0),(0,511),(255,0,0),1)
cv2.imshow('test',img)
cv2.resizeWindow('test',512,10)
cv2.waitKey(0)
cv2.destroyAllWindows()