import cv2
import numpy as np
from scipy import io
import  lda

X=io.loadmat('b.mat')['matrix'].astype(int)
model=lda.LDA(n_topics=2,n_iter=1500,random_state=1)
model.fit(X)
io.savemat('c.mat', {'matrix': model.doc_topic_})
print(model.doc_topic_)