import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_path = './signs_vehicles_xygrad.png'

image = mpimg.imread(img_path)
thresh = (180, 255)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
binary = np.zeros_like(gray)
binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1

def show(original_image, new_image):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(new_image, cmap='gray')
    ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()

#show(image, binary)

# Plotting thresholded images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Original')
ax1.imshow(image)

ax2.set_title('Thresholded Grad. Dir.')
ax2.imshow(binary, cmap='gray')
plt.show()
