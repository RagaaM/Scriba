import numpy as np
import cv2
from matplotlib import pyplot as plt


    


# img = cv2.imread('Dataset/Automated/Raw/3/030000_D35.png')
img = cv2.imread('Dataset/Automated/Raw/3/030001_V28.png')
# img = cv2.imread('Dataset/Automated/Raw/3/030010_G39.png')
# img = cv2.imread('Dataset/Automated/Raw/3/030013_UNKNOWN.png')
# img = cv2.imread('Dataset/Automated/Raw/3/030014_N35.png')


gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gaussian_blur = cv2.GaussianBlur(gray_image, (5,5), 1.5)
ret, thresh1 = cv2.threshold(gaussian_blur, 0, 255, cv2.THRESH_BINARY_INV + 
                                            cv2.THRESH_OTSU)


plt.imshow(thresh1, cmap='gray')
plt.axis('off')
plt.show()