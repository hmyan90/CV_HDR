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
    os.mkdir('part3_plot')
except:
    pass
    
# Read images
img_dir="Part2_fig/"
three_files = ["3000.jpeg", "1000.jpeg", "500.jpeg"]
time = [1./3000, 1./1000, 1./500]
a_list = [1, 3, 6]
g_channel = [2.061331, 2.786699, 2.768208] # get from Part1

height, width = 2448, 2448
imgs = []
for i in range(0, len(three_files)):
    imgs.append(cv2.imread(img_dir+three_files[i]))

def alg1():
    print("Start alg1")
    HDR_img = np.zeros((height, width, 3))
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
    HDR_img = np.zeros((height, width, 3))
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

def plot_hist(img, alg=1):

    color = ('b','g','r')
    for channel, col in enumerate(color):
        img_channel=img[:,:,channel]
        plt.subplot(2,2,channel+1)
        _max = int(pow(255, g_channel[channel])) + 1
        plt.hist(img_channel.ravel(),bins=25,range=[0,_max],color=col)
        plt.title('%s Channel' %color[channel].upper())
    
    plt.savefig('part3_plot/alg%d_histogram.jpg' %(alg))
    plt.gcf().clear()

hdr1 = alg1()
print hdr1
hdr2 = alg2()
plot_hist(hdr1,alg=1)
plot_hist(hdr2,alg=2)

