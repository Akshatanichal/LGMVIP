#!/usr/bin/env python
# coding: utf-8
# # LetsGrowMore: Data Science
# ## Task01 (Beginner Level Task): Image to Pencil Sketch By Using Python
# ### Name of Intern:Akshata Naganath Nichal
import cv2 
import numpy as np
import matplotlib.pyplot as plt
def show_image(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
img=cv2.imread("C:/Users/91986/Downloads/21-color-pencil-drawing-bird.jpg")
show_image(img)
plt.figure(num=1,figsize=(8,8))
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
invert_img=cv2.bitwise_not(gray_img)
blur_img=cv2.GaussianBlur(invert_img,(111,111),0)
invblur_img=cv2.bitwise_not(blur_img)
sketch_img=cv2.divide(gray_img,invblur_img,scale=256)
cv2.imwrite('Sketch.jpg',sketch_img)
cv2.imshow('Sketch image',sketch_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.figure(figsize=(14,8))
plt.subplot(1,2,1)
plt.title('Original image',size=18)
plt.imshow(img)
plt.axis('off')
plt.subplot(1,2,1)
plt.title('Sketch',size=18)
rgb_sketch=cv2.cvtColor(sketch_img,cv2.COLOR_BGR2RGB)
plt.imshow(rgb_sketch)
plt.axis('off')
plt.show()
