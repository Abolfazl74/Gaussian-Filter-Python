import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

img = cv.imread('lenna.jpg', cv.IMREAD_GRAYSCALE)
height, width= img.shape[:2]

plt.figure(0)
plt.title("Input image")
plt.imshow(img, cmap=cm.gray)


d_width = width + 2
d_height = height + 2
result = np.zeros((d_height,d_width))
applied_img = np.zeros_like(img)

xx = (d_width - width) // 2
yy = (d_height - height) // 2

result[yy:yy+height, xx:xx+width] = img

plt.figure(1)
plt.title("Result of zero-padding")
plt.imshow(result, cmap=cm.gray)

kernel = np.array(([1,2,1]
                  ,[2,4,2]
                  ,[1,2,1]))

cdf = np.zeros_like(kernel)

for i in range(yy, height):
        for j in range(xx, width):
            cdf[0,0] = result[i-1, j-1] * kernel[0,0]
            cdf[0,1] = result[i, j-1] * kernel[0,1]
            cdf[0,2] = result[i+1, j-1] * kernel[0,2]
            cdf[1,0] = result[i-1, j] * kernel[1,0]
            cdf[1,1] = result[i, j] * kernel[1,1]
            cdf[1,2] = result[i+1, j] * kernel[1,2]
            cdf[2,0] = result[i-1, j+1] * kernel[2,0]
            cdf[2,1] = result[i, j+1] * kernel[2,1]
            cdf[2,2] = result[i+1, j+1] * kernel[2,2]
            
            applied_img[i, j] = np.sum(cdf) / 16
            

plt.figure(2)
plt.title("Final image")
plt.imshow(applied_img, cmap=cm.gray)
plt.show()     
