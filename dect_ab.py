import cv2
import numpy as np
from scipy import io
import  lda

X=io.loadmat('c.mat')['matrix'].astype(int)