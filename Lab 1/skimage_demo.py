import matplotlib.pyplot as plt 

from skimage import data,filters

image = data.coins()
# ... or any other NumPy array!

edges = filters.sobel(image)

plt.imshow(image, cmap='gray')
print('image.shape={} imahe.max()={}'.format(image.shape, image.max()))
plt.show()



plt.imshow(edges, cmap='gray')
plt.write('aa.jpg')

plt.show()