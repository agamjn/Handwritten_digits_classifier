#necessary imports

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

#change directory to working directory
os.chdir('location of dir')
img=cv2.imread('image',0)
#converting to binary
(thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
thresh = 127
im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
print (im_bw.shape)
#reshaping into desired 28*28 pixels
resize=cv2.resize(im_bw,(28,28))
plt.imshow(resize)
plt.title("desired")
print (resize.shape)
dim_image=np.prod(resize.shape)
test_image = resize.reshape(1,dim_image)
test_image=255-test_image
print (test_image.shape)
