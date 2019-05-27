#Author: Hélène Urien, 28/04/2019
#Try to be Pep8 compliant !  
#https://www.python.org/dev/peps/pep-0008/

#System imports
from __future__ import print_function
from __future__ import division
import os

#Third part import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from PIL import Image
from scipy import ndimage
from skimage import feature
from skimage.feature import peak_local_max

#Get the current script directory
script_dir = os.path.dirname(__file__)
im_dir = os.path.join(script_dir, "images")

###############################################################################
#I Otsu method
###############################################################################

#Define the Otsu thresholding function 
def otsu(arr):
    
    #Convert intensity values into int values
    arr = arr.astype(int)
    
    #List of intensity values
    vals = list(np.unique(arr))  
    vals_hist = list(np.copy(vals))   
    vals_hist.append(max(vals) + 1)

    #Compute the image histogram
    hist, bin_edges = np.histogram(arr, bins=vals_hist)

    #Normalize the histogram
    p_i = hist / hist.sum()

    #Test each threshold value partionning the histogram into 2 classes
    sigmas_b2 = []
    for cnt, threshold_val in enumerate(vals[0:len(vals) - 1]):
        w0 = p_i[0:cnt + 1].sum()
        w1 = p_i[cnt + 1:].sum()
        sigma_b2 = ((w0 * (vals * p_i).sum() - (vals[0:cnt + 1] * p_i[0:cnt + 1]).sum()) ** 2) / (w0 * w1)
        sigmas_b2.append(sigma_b2)
    threshold_val = vals[np.argmax(sigmas_b2)]
    return threshold_val

#Test the Otsu method on the red channel of two images

for imname in ["hibiscus.bmp", "coins.jpg"]:
    #Load the color image   
    im_arr = Image.open(os.path.join(im_dir, imname))
    im_arr = np.array(im_arr)

    #Retain only the red channel
    glim_arr = im_arr[:,:,0]
    
    threshold_val = otsu(glim_arr)
    tglim_arr = (glim_arr > threshold_val).astype(int)

    #Plot the image, its histogramm and the contours
    hist, bin_edges = np.histogram(glim_arr, bins=50)
    centers = bin_edges[:len(bin_edges) - 1] + 0.5 *(bin_edges[1:] - bin_edges[0:len(bin_edges) - 1])
    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    fig.subplots_adjust(hspace=10, wspace=0.1)
    ax[0].set_title("Red channel")
    ax[0].imshow(glim_arr, cmap ="gray")
    ax[0].axis("off")
    ax[1].set_title("Threshold image")
    ax[1].imshow(tglim_arr, cmap ="gray")
    ax[1].axis("off")
    ax[2].plot(centers, hist, linewidth=0.5, linestyle="--");
    ax[2].set_ylim([hist.min(), hist.max()])
    ax[2].set_xlim([centers.min(), centers.max()])
    ax[2].axvline(x=threshold_val, linewidth=1, color='r')
    ax[2].set_xlabel("Intensity value")
    ax[2].set_ylabel("Count")
    ax[2].set_title("Histogram of red channel")
    fig.savefig("Ex1_{0}.png".format(imname.split(".")[0]))


###############################################################################
#II Region growing
###############################################################################

stop
#Load the color image   
im_arr = Image.open(os.path.join(im_dir, "hibiscus.bmp"))
im_arr = np.array(im_arr)


#Take the red channel
glim_arr =im_arr[:,:,0]
glim_arr = glim_arr / np.max(glim_arr)

#Downsample to reduce computationnal time...
glim_arr = glim_arr[::4, ::4]
          
#Image dimension
sx, sy = glim_arr.shape

#Initial seed
x0 = 50
y0 = 50

#Try two homogeneity criteria
for epsilon, criterion in zip([0.1, 0.2], ["standard_variation", "contrast"]):

    #Initialize the segmentation array
    seg_arr = np.zeros((sx, sy))
    seg_arr[x0, y0] = 1
    converge_score = 1
    
    while converge_score != 0:
        
        #Compute the neighbourhood of the current segmented area, not taking into
        #account pixels yet in the segmented area

        #a)Voxel-wise (8-connectivity)
        neigh = np.zeros((sx, sy))
        [x, y] = np.where(seg_arr == 1)
        for x_i, y_i in zip(x, y):
            
            if x_i == sx - 1:
                mx = 1
            else:
                mx = 2
            if y_i == sy - 1:
                my = 1
            else:
                my = 2
            test = seg_arr[x_i - 1:x_i + mx, y_i - 1: y_i + my]
            local_neigh = np.ones((test.shape[0], test.shape[1]))
            local_neigh[seg_arr[x_i - 1:x_i + mx, y_i - 1: y_i +2] == 1] = 0
            local_neigh = np.logical_or(local_neigh,
                                        neigh[x_i - 1:x_i +mx, y_i - 1: y_i +2])
            local_neigh = local_neigh.astype(int)
            neigh[x_i - 1:x_i +mx, y_i - 1: y_i +2] = local_neigh

        #b) Morphological gradient (8-connectivity)
        #se = np.ones((3, 3))
        #neigh = ndimage.binary_dilation(seg_arr, structure=se).astype(int) - seg_arr    

        #c)Distance transform (4-connectivity)
        #neigh = np.logical_and(ndimage.distance_transform_edt(1 - seg_arr) <= 1,
        #                       seg_arr == 0).astype(int)

        #For each voxel of the segmented region neighbourhood
        [x, y] = np.where(neigh == 1)
        new_seg_arr = np.copy(seg_arr)
        for x_i, y_i in zip(x, y):
            potential_seg_arr = np.copy(new_seg_arr)
            potential_seg_arr[x_i, y_i] = 1
            if criterion == "standard_variation":
                potential_new_sigma = np.std(glim_arr[potential_seg_arr == 1])
            elif criterion == "contrast":
                potential_new_sigma = max(glim_arr[potential_seg_arr == 1]) - min(glim_arr[potential_seg_arr == 1])
            if potential_new_sigma < epsilon:
                new_seg_arr[x_i, y_i] = 1       
        previous_seg_arr = np.copy(seg_arr)        
        seg_arr = np.copy(new_seg_arr)
        converge_score = np.count_nonzero(new_seg_arr - previous_seg_arr)

    #Save the result
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    fig.subplots_adjust(hspace=10, wspace=0.1)
    ax.set_title("Grayscale image")
    ax.imshow(glim_arr, cmap="gray")    
    ax.contour(seg_arr, colors="r", linewidths=2, levels=[0.5, 1])
    ax.axis("off")
    fig.savefig("Ex3_{0}.png".format(criterion))

###############################################################################
#III Hough transform
###############################################################################
#Load the color image   
im_arr = Image.open(os.path.join(im_dir, "coins.jpg"))
im_arr = np.array(im_arr)

#Convert it into grayscale
glim_arr = np.mean(im_arr, 2)
glim_arr = glim_arr / np.max(glim_arr)

#Image croping
glim_arr = glim_arr[100:2500, 500:3150]

#Downsample
glim_arr = glim_arr[::5, ::5]
sx, sy = glim_arr.shape

edges_arr = feature.canny(glim_arr, sigma=2)

#a) Detect circles
sx, sy = edges_arr.shape
x_e, y_e = np.where(edges_arr > 0)
print(len(x_e))

theta_vals = range(0, 361)
r_vals = range(12, 21)
acc = np.zeros((sx, sy, 1 + np.max(r_vals))).astype(int)
cnt=0
for x, y in zip(x_e, y_e):
    cnt+=1
    print(cnt)
    for r in r_vals:
        for theta in theta_vals:
            a = int(x - r * np.cos((theta * np.pi) / 180))
            b = int(y - r * np.sin((theta * np.pi) / 180))
            if a <= sx - 1 and a >= 0 and b <= sy - 1 and b >=0:
                acc[a, b, r] += 1

#Significant maximum
circles_arr = np.zeros((sx, sy)).astype(int)
centers_arr = np.zeros((sx, sy)).astype(int)
rs_arr = np.zeros((sx, sy)).astype(int)
y, x = np.meshgrid(range(0, sy), range(0, sx))
for r in r_vals:
    #Normalize each accumator slice
    acc_2D = acc[:, :, r] / np.max(acc[:, :, r])  # ##
    if np.count_nonzero(acc_2D) > 0:
        #Get each pixel with intensity value greater than 60% of the maximal intensity
        #of the accumulator slice array
        [a, b] = np.where(acc_2D >= 0.6)
        for a_i, b_i in zip(a, b):
            #Draw the circle around each coin center
            circle_arr = ((x - a_i) * (x - a_i) + (y - b_i) * (y - b_i) <= r * r).astype(int)
            #Draw each coin center
            centers_arr[a_i, b_i] = 1
            #Update the circle array
            circles_arr[circle_arr == 1] = 1
            rs_arr[a_i, b_i] = r

#Plot the result            
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
fig.subplots_adjust(hspace=10, wspace=0.1)
ax.axis("off")
ax.imshow(glim_arr, cmap ="gray")
if np.count_nonzero(centers_arr) > 0 :
    ax.contour(centers_arr, colors="r", levels=[0.5, 1], linewidth = 1)
if np.count_nonzero(circles_arr) > 0 :
    ax.contour(circles_arr, colors="b", levels=[0.5, 1], linewidth = 1)
fig.savefig("Ex2_detect_circles.png")

#Some coins have several centers: clean them !
#Remove connected components of of area less than 10 voxels ** 2
label_arr, nb_ccs = ndimage.label(circles_arr, structure=np.ones((3, 3)))
new_circles_arr = np.zeros((sx, sy)).astype(int)
new_centers_arr = np.zeros((sx, sy)).astype(int)
for label_id in range(1, nb_ccs + 1):
    coin_arr = (label_arr == label_id).astype(int)
    [a, b] = np.where(np.logical_and(coin_arr == 1, centers_arr == 1))
    if len(a) == 1:
        new_circles_arr[coin_arr == 1] = 1
        new_centers_arr[a[0], b[0]] = 1
    else:
        scores = []
        rs = []
        for a_i, b_i in zip(a, b):
            #Draw the circle around each coin center
            r = rs_arr[a_i, b_i]
            rs.append(r)
            circle_arr = ((x - a_i) * (x - a_i) + (y - b_i) * (y - b_i) <= r * r).astype(int)
            area = np.count_nonzero(circle_arr)
            score = (np.max(glim_arr[circle_arr == 1]) - np.min(glim_arr[circle_arr == 1])) / area
            scores.append(score)
        print(scores)
        print(rs)
        print('\n')
        ind_r = np.argmin(scores)
        a_i = a[ind_r]
        b_i = b[ind_r]
        r = rs_arr[a_i, b_i]
        circle_arr = ((x - a_i) * (x - a_i) + (y - b_i) * (y - b_i) <= r * r).astype(int)
        new_circles_arr[circle_arr == 1] = 1
        new_centers_arr[a_i, b_i] = 1

circles_arr = np.copy(new_circles_arr)
centers_arr = np.copy(new_centers_arr)

#Plot the result            
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
fig.subplots_adjust(hspace=10, wspace=0.1)
ax.axis("off")
ax.imshow(glim_arr, cmap ="gray")
if np.count_nonzero(centers_arr) > 0 :
    ax.contour(centers_arr, colors="r", levels=[0.5, 1], linewidth = 1)
if np.count_nonzero(circles_arr) > 0 :
    ax.contour(circles_arr, colors="b", levels=[0.5, 1], linewidth = 1)
fig.savefig("Ex2_detect_circles_clean.png")
