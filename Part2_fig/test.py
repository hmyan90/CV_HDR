import numpy as np
import cv2


name = "6667"
cnt = 0
img = cv2.imread(name+".JPG")
for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        for k in range(0, img.shape[2]):
            if img[i][j][k] == 255:
                cnt += 1
                print i,j,k,img[i][j][k]

print cnt
