import numpy as np
import cv2 as cv

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

img = cv.imread('messi5.jpg', 1)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print('img.shape={}'.format(img.shape))

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

plt.savefig('foo.png')
plt.savefig('foo.pdf')