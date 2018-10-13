#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:42:39 2018

@author: yb
"""

#%% LIBs
from  skimage import io, morphology
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
#%% folder
data_path  = 'D:/All in one/PhD data/Dataset/data/013/images/'

#%% code
im  = io.imread(os.path.join(data_path, '01.tiff'))

im_G  = im[:,:,2]
plt.figure()
io.imshow(im)

plt.figure()
io.imshow(im_G)

#%%
im_bin  = (1-(im_G> 230) * 1).astype('uint8')

kernel = np.ones((10,10), np.uint8)
im_bin_dilated = cv2.erode(im_bin, kernel=kernel, iterations  = 1)

plt.figure()
io.imshow(im_bin_dilated,cmap = 'gray')

#%%OTsu thresholding

# Otsu's thresholding
ret2,th2 = cv2.threshold(im_G,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
th2  = 255-th2 # complement
#stel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
stel = morphology.disk(50)
im_dilte  = cv2.dilate(th2, kernel=stel, iterations = 1)

label_img = label(im_dilte, connectivity=im_dilte.ndim)
props = regionprops(label_img)

labeled_reg_areas  = np.array([ props[i].area for i in range (len(props))])

S  = np.sum(labeled_reg_areas)
Index = np.argmax(labeled_reg_areas)

label_image_cleaned  = (label_img == Index + 1) * 1

X  = np.repeat(np.expand_dims(label_image_cleaned,axis = -1), 3, axis  = 2)
Y  = X.astype('int8')*im

Y [Y==0] = 220
plt.figure()
io.imshow(Y.astype('uint8'))

plt.figure()
io.imshow(im)
