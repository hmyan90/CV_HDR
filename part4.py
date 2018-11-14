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
time = [1./4425, 1./1000, 1./250]
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
    for c in range(0, 3):
        img0 = np.power(imgs[0][:,:,c], g_channel[c])
        img1 = np.power(imgs[1][:,:,c], g_channel[c])
        img2 = np.power(imgs[2][:,:,c], g_channel[c])

        for h in range(0, height):
            for w in range(0, width):
                if imgs[2][h][w][c] < 255: # use third image
                    HDR_img[h][w][c] = img2[h][w]/a_list[2]
                elif imgs[1][h][w][c] < 255: # use second image
                    HDR_img[h][w][c] = img1[h][w]/a_list[1]
                else: # use first image
                    HDR_img[h][w][c] = img0[h][w]/a_list[0]
    return HDR_img

def alg2():
    print("Start alg2")
    HDR_img = np.zeros((height, width, 3), dtype='float32')
    for c in range(0, 3):
        img0 = np.power(imgs[0][:,:,c], g_channel[c])
        img1 = np.power(imgs[1][:,:,c], g_channel[c])
        img2 = np.power(imgs[2][:,:,c], g_channel[c])

        for h in range(0, height):
            for w in range(0, width):
                if imgs[2][h][w][c] < 255: # use all three images
                    HDR_img[h][w][c] = (img2[h][w]/a_list[2] + img1[h][w]/a_list[1] + img0[h][w]/a_list[0])/3.
                elif imgs[1][h][w][c] < 255: # use second and first image
                    HDR_img[h][w][c] = (img1[h][w]/a_list[1] + img0[h][w]/a_list[0])/2.
                else: # use first image
                    HDR_img[h][w][c] = img0[h][w]/a_list[0]
    return HDR_img

def tonemap(hdr, alg=1):
    tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    ldrDrago = tonemapDrago.process(hdr)
    ldrDrago = 3 * ldrDrago
    cv2.imwrite("part4_plot/hdr%d.jpg" %alg, ldrDrago * 255)

tonemap(alg1(), alg=1)
tonemap(alg2(), alg=2)
