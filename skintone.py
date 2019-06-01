import cv2
import numpy as np

img = cv2.imread(r"D:\Intern\test images\test13.jpg")
#upstate = cv2.resize(upstate,(250,400))
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("Original",img)
lower = np.array([0, 48, 80], dtype=np.uint8)
upper = np.array([20, 255, 255], dtype=np.uint8)
# get mask of pixels that are in blue range
img_hsv = cv2.GaussianBlur(img_hsv, (13,13), 0)
mask = cv2.inRange(img_hsv, lower, upper)

cv2.imshow("mask",cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))

# convert single channel mask back into 3 channels
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

# perform bitwise and on mask to obtain cut-out image that is not blue
masked_img = cv2.bitwise_and(img, mask_rgb)
cv2.imshow(" masked image",masked_img)
cv2.waitKey(0)
cv2.destroyAllWindows()