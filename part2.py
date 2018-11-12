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
three_files = ["3000.jpeg", "1000.jpeg", "500.jpeg"]
time = [1./3000, 1./1000, 1./500]
a_list = [1, 3, 6]
g_channel = [2.061331, 2.786699, 2.768208] # get from Part1
colors = ['B', 'G', 'R']

# Plot
for i in range(0, len(three_files)):
    img=cv2.imread(img_dir+three_files[i])
    for channel, col in enumerate(colors):
        img_channel=img[:,:,channel]
        b_prime_g = np.power(img_channel, g_channel[channel])
        plt.subplot(2,2,channel+1)
        _max = int(pow(255,g_channel[channel]))+1
        plt.hist(b_prime_g.ravel(),bins=25,range=[0, _max])
        plt.title('%s Channel' %col)

    plt.savefig('part2_plot/B\'_vs_T_image_%s_channel_%s.jpg' %(three_files[i],col))
    plt.gcf().clear()

    if i > 0:
        a = a_list[i]
        for channel, col in enumerate(colors):
            img_channel=img[:,:,channel]
            b_prime_g = np.power(img_channel, g_channel[channel])
            b_prime_g_div_a=np.divide(b_prime_g, a)
            plt.subplot(2,2,channel+1)
            _max = int(pow(255,g_channel[channel])/a)+1
            plt.hist(b_prime_g_div_a.ravel(),bins=25,range=[0, _max])
            plt.title('%s Channel' %col)
        
        plt.savefig('part2_plot/B\'_div_a_vs_T_image_%s_channel_%s.jpg' %(three_files[i],col))
        plt.gcf().clear()

