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
nfiles = ["0060", "0090", "0125", "0180", "0250", "0350", "0500", "0750", "1000", "1500", "2000", "3000"]
rgb=list()
for fl in nfiles:
    img=cv2.imread(img_dir+fl+'.JPG')
    height, width=img.shape[:2]
    size = 100
    patch=img[(height/2-size):(height/2+size),(width/2-size):(width/2+size)] # [2*size, 2*size, 3]
    cv2.imwrite("part1_plot/%s.png" %fl, img)
    patch_rgb=patch.mean(axis=(0,1))
    rgb.append(patch_rgb)

rgb_value=array(rgb) # [number_of_image, 3]
print('The RBG of each image is ', rgb_value)

# Calculate B'(T), logB', B'^g 
T=[1./60, 1./90, 1./125, 1./180, 1./250, 1./350, 1./500, 1./750, 1./1000, 1./1500, 1./2000, 1./3000]
logT=np.log2(T)
colors = ['b', 'g', 'r']

g = []
B_primes = []
logB_primes = []
lines = []
B_prime_g_s = []

for channel, col in enumerate(colors):
    # calculate B'(T) of each channel
    B_prime = rgb_value[:, channel]
    B_primes.append(B_prime)

    # calculate logB' of each channel 
    logB_prime = np.log2(B_prime)
    logB_primes.append(logB_prime)

    # calculate linear regression of logB' vs logT
    slope, intercept, rvalue, p_value, std_err = stats.linregress(logT, logB_prime)
    g = 1./slope 
    print('Channel %s: g = %f, a= %f, b = %f' %(col.upper(), g, slope, intercept))
    line = logT*slope+intercept
    lines.append(line)

    # calculate B'^g 
    B_prime_g = np.power(B_prime, g)
    B_prime_g_s.append(B_prime_g)

# Plot B'(T) as a function of T 
for channel, col in enumerate(colors):
    plt.subplot(2,2,channel+1)
    plt.plot(T, B_primes[channel], color=col)
    plt.xlabel('T(s)')
    plt.ylabel('B\'')
    plt.title('%s Channel' %col.upper())

plt.gcf().set_size_inches(18.5, 10.5)
plt.savefig('part1_plot/B\'_vs_T.jpg')
plt.gcf().clear()

# Plot logB' as a function of logT 
for channel, col in enumerate(colors):
    plt.subplot(2,2,channel+1)
    plt.plot(logT, logB_primes[channel], color=col)
    plt.plot(logT, lines[channel], color='C1')        
    plt.xlabel('log T(s)')
    plt.ylabel('log B\'')
    plt.title('%s Channel' %col.upper())

plt.gcf().set_size_inches(18.5, 10.5)
plt.savefig('part1_plot/logB\'_vs_logT.jpg')
plt.gcf().clear()

# Plot B'^g as a function of T
for channel, col in enumerate(colors):
    plt.subplot(2,2,channel+1)
    plt.plot(T, B_prime_g_s[channel], color=col)  
    plt.xlabel('T(s)')
    plt.ylabel('B\'^g')
    plt.title('%s Channel' %col.upper())

plt.gcf().set_size_inches(18.5, 10.5)
plt.savefig('part1_plot/B\'^g_vs_T.jpg')
plt.gcf().clear()
