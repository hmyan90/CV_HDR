import numpy as np
import cv2
import os
import glob
from numpy import array
import matplotlib.pyplot as plt
from scipy import stats
import math

try:
    os.mkdir('part1_plot')
except:
    pass
    
# Read images, crop middle part
img_dir="Part1_fig/"
nfiles = ["020", "030", "045", "060", "090", "125", "180", "250", "350", "500", "750"] #, "1000", "1500", "2000", "3000"]
rgb=list()
for fl in nfiles:
    img=cv2.imread(img_dir+fl+'.jpeg')
    height, width=img.shape[:2]
    size = 500
    patch=img[(height/2-size):(height/2+size),(width/2-size):(width/2+size)] # [2*size, 2*size, 3]
    patch_rgb=patch.mean(axis=(0,1))
    rgb.append(patch_rgb)

rgb_value=array(rgb) # [number_of_image, 3]
print('The RBG of each image is ', rgb_value)

# Plot below
T=[1./20,1./30,1./45,1./60,1./90,1./125,1./180,1./250,1./350,1./500, 1./750] #, 1./1000, 1./1500, 1./2000, 1./3000]
logT=np.log2(T)
colors = ['B', 'G', 'R']
for channel, col in enumerate(colors):
    # B'(T) as a function of T 
    B_prime = rgb_value[:, channel]
    plt.plot(T, B_prime)
    plt.xlabel('T(s)')
    plt.ylabel('B\'')
    plt.title('%s Channel' %col)
    plt.savefig('part1_plot/B\'_vs_T_%s_channel.jpg' %col)
    plt.gcf().clear()

    # logB' as a function of logT 
    logB_prime=np.log2(B_prime)
    slope, intercept, rvalue, p_value, std_err = stats.linregress(logT,logB_prime)
    g = 1./slope 
    print('Channel %s: g = %f, a= %f, b = %f' %(colors[channel], g, slope, intercept))
    line=logT*slope+intercept
    plt.plot(logT,logB_prime)
    plt.plot(logT,line)        
    plt.xlabel('log T(s)')
    plt.ylabel('log B\'')
    plt.title('%s Channel' %col)
    plt.savefig('part1_plot/logB\'_vs_logT_%s_channel.jpg' %col)
    plt.gcf().clear()

    # B'^g as a function of T
    B_prime_g = np.power(B_prime, g)
    plt.plot(T,B_prime_g)      
    plt.xlabel('T(s)')
    plt.ylabel('B\'$^{g}$')
    plt.title('%s Channel' %col)
    plt.savefig('part1_plot/B\'$^{g}$_vs_T_%s_channel.jpg' %col)
    plt.gcf().clear()
    
