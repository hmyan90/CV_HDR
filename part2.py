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
    os.mkdir('part2_plot')
except:
    pass
    
# Read images
img_dir="Part2_fig/"
three_files = ["4425.JPG", "1000.JPG", "0350.JPG"]
time = [1./4425, 1./1000, 1./350]
a_list = [1., 4425./1000, 4425./350]
g_channel = [2.465, 2.510, 2.518] # get from Part1
colors = ['b', 'g', 'r']

# Plot
for i in range(0, len(three_files)):
    img=cv2.imread(img_dir+three_files[i])
    for channel, col in enumerate(colors):
        img_channel=img[:,:,channel]
        b_prime_g = np.power(img_channel, g_channel[channel])
        plt.subplot(2,2,channel+1)
        _max = int(pow(255,g_channel[channel]))+1
        plt.hist(b_prime_g.ravel(),bins=25,range=[0, _max],color=col.lower())
        plt.title('%s Channel' %col.upper())

    plt.gcf().set_size_inches(18.5, 10.5)
    plt.savefig('part2_plot/B\'_vs_T_image_%s.jpg' %(three_files[i]))
    plt.gcf().clear()

    if i > 0:
        a = a_list[i]
        for channel, col in enumerate(colors):
            img_channel=img[:,:,channel]
            b_prime_g = np.power(img_channel, g_channel[channel])
            b_prime_g_div_a=np.divide(b_prime_g, a)
            plt.subplot(2,2,channel+1)
            _max = int(pow(255,g_channel[channel])/a)+1
            plt.hist(b_prime_g_div_a.ravel(),bins=25,range=[0, _max], color=col.lower())
            plt.title('%s Channel' %col.upper())
        
        plt.gcf().set_size_inches(18.5, 10.5)
        plt.savefig('part2_plot/B\'_div_a_vs_T_image_%s.jpg' %(three_files[i]))
        plt.gcf().clear()
