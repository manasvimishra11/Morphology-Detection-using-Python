#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import skimage.filters
import skimage.color
import skimage.io
import matplotlib.pyplot as plt


bacteria_image = skimage.io.imread("colonies-01.jpg")

# display the image
fig, ax = plt.subplots()
plt.imshow(bacteria_image)
plt.show()
gray_bacteria = skimage.color.rgb2gray(bacteria_image)
# display the gray image
fig, ax = plt.subplots()
plt.imshow(gray_bacteria, cmap='gray')
plt.show()
#create a histogram 
blurred_image = skimage.filters.gaussian(gray_bacteria, sigma=1.0)
histogram, bin_edges = np.histogram(blurred_image, bins=256, range=(0.0, 1.0))
fig, ax = plt.subplots()
plt.plot(bin_edges[0:-1], histogram)
plt.title("Graylevel histogram")
plt.xlabel("gray value")
plt.ylabel("pixel count")
plt.xlim(0, 1.0)
plt.show()
#mask shows us where the colonies are therefore we create a mask
mask = blurred_image < 0.2
fig, ax = plt.subplots()
plt.imshow(mask, cmap='gray')
plt.show()

#next we perform component connected analysis 
labeled_image, count = skimage.measure.label(mask, return_num=True)
print(count)

#finally we create a summary image of count of colonies
# color each of the colonies a different color
colored_label_image = skimage.color.label2rgb(labeled_image, bg_label=0)
# give our grayscale image rgb channels, so we can add the colored colonies
summary_image = skimage.color.gray2rgb(gray_bacteria)
summary_image[mask] = colored_label_image[mask]

# plot overlay
fig, ax = plt.subplots()
plt.imshow(summary_image)
plt.show()


# In[2]:


#we define the above process into a reusable function
def count_colonies(image_filename):
    bacteria_image = skimage.io.imread(image_filename)
    gray_bacteria = skimage.color.rgb2gray(bacteria_image)
    blurred_image = skimage.filters.gaussian(gray_bacteria, sigma=1.0)
    mask = blurred_image < 0.2
    labeled_image, count = skimage.measure.label(mask, return_num=True)
    print(f"There are {count} colonies in {image_filename}")

    colored_label_image = skimage.color.label2rgb(labeled_image, bg_label=0)
    summary_image = skimage.color.gray2rgb(gray_bacteria)
    summary_image[mask] = colored_label_image[mask]
    fig, ax = plt.subplots()
    plt.imshow(summary_image)

for image_filename in ["colonies-01.jpg", "colonies-02.jpg", "colonies-03.jpg"]:
    count_colonies(image_filename)


# In[ ]:




