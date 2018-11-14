import numpy as np
import cv2
import os
import glob
from numpy import array
import matplotlib.pyplot as plt
from scipy import stats
import math
import matplotlib.mlab as mlab

try:
    os.mkdir('part4_plot')
except:
    pass
    
# Read images
img_dir="Part2_fig/"
three_files = ["4425.JPG", "1000.JPG", "0350.JPG"]
time = [1./4425, 1./1000, 1./350]
a_list = [1., 4425./1000, 4425./350]
g_channel = [2.465, 2.510, 2.518] # get from Part1
colors = ['b', 'g', 'r']

height, width = 3264, 2448
imgs = []
for i in range(0, len(three_files)):
    imgs.append(cv2.imread(img_dir+three_files[i]))

def alg1():
    print("Start alg1")

    HDR_img = np.zeros((height, width, 3), dtype='float32')
    img0, img1, img2 = imgs[0], imgs[1], imgs[2]

    for h in range(0, height):
        for w in range(0, width):
            # if img2[h][w][0] < 255 and img2[h][w][1] < 255 and img2[h][w][2] < 255:
            if img2[h][w].max() < 255:
                for c in range(0, 3):
                    HDR_img[h][w][c] = np.power(img2[h][w][c], g_channel[c])/a_list[2]
            # elif img1[h][w][0] < 255 and img1[h][w][1] < 255 and img1[h][w][2] < 255:
            elif img2[h][w].max() < 255:
                for c in range(0, 3):
                    HDR_img[h][w][c] = np.power(img1[h][w][c], g_channel[c])/a_list[1]       
            else:
                for c in range(0, 3):
                    HDR_img[h][w][c] = np.power(img0[h][w][c], g_channel[c])/a_list[0]

    HDR_img = np.array(HDR_img, dtype='float32')
    return HDR_img

def alg2():
    print("Start alg2")
    HDR_img = np.zeros((height, width, 3), dtype='float32')
    img0, img1, img2 = imgs[0], imgs[1], imgs[2]

    for h in range(0, height):
        for w in range(0, width):
            if img2[h][w].max() < 255:
                for c in range(0, 3):
                    i2 = np.power(img2[h][w][c], g_channel[c])/a_list[2] 
                    i1 = np.power(img1[h][w][c], g_channel[c])/a_list[1] 
                    i0 = np.power(img0[h][w][c], g_channel[c])/a_list[0]
                    HDR_img[h][w][c] = (i2+i1+i0)/3.
            elif img2[h][w].max() < 255:
                for c in range(0, 3):
                    i1 = np.power(img1[h][w][c], g_channel[c])/a_list[1] 
                    i0 = np.power(img0[h][w][c], g_channel[c])/a_list[0]
                    HDR_img[h][w][c] = (i1+i0)/2.     
            else:
                for c in range(0, 3):
                    i0 = np.power(img0[h][w][c], g_channel[c])/a_list[0]
                    HDR_img[h][w][c] = i0

    HDR_img = np.array(HDR_img, dtype='float32')
    return HDR_img

def tonemap(hdr, alg=1):
    tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    ldrDrago = tonemapDrago.process(hdr)
    ldrDrago = 3 * ldrDrago
    cv2.imwrite("part4_plot/hdr%d.jpg" %alg, ldrDrago * 255)

tonemap(alg1(), alg=1)
tonemap(alg2(), alg=2)

