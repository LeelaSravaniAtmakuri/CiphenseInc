import cv2
import numpy as np
from PIL import Image
import statistics
from math import sqrt
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

img = cv2.imread(r"D:\Intern\test images\test13.jpg")
#img = cv2.resize(img,(250,400))
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

def calcAvgPixel(im):
    pixels = list(im.getdata())
    #print(pixels)
    pixel_list = [x for sets in pixels for x in sets]
    #print(pixel_list)
    pixel_red = pixel_list[::3]
    pixel_green = pixel_list[1::3]
    pixel_blue = pixel_list[2::3]
    red = int(statistics.mean(np.array(pixel_red)))
    green = int(statistics.mean(np.array(pixel_green)))
    blue = int(statistics.mean(np.array(pixel_blue)))
    return (red, green, blue);
'''
def euclideanDist(r1,g1,b1,r2,g2,b2):
    return (sqrt((r2-r1)**2 +(g2-g1)**2 + (b2-b1)**2))
'''
face_detect = cv2.CascadeClassifier(r"D:\Anaconda\Library\etc\haarcascades\haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)			  # Convert color image to gray image
face = face_detect.detectMultiScale(gray, 1.2, 5)
for (x,y,w,h) in face:
    masked_img_face = masked_img[y:y+h, x:x+w]
    x_left_start = x
    y_left_start = y
    x_left_end = int(x+w/2)
    y_left_end = int(y+h)
    x_right_start = x_left_end
    y_right_start = y
    x_right_end = x+w
    y_right_end = y+h
    left_half = masked_img[y_left_start:y_left_end, x_left_start:x_left_end]
    right_half = masked_img[y_right_start:y_right_end, x_right_start:x_right_end]


cv2.imshow("face",masked_img_face)
cv2.imshow("left face",left_half)
cv2.imshow("right face",right_half)

cv2.imwrite("left face.jpg",left_half)
cv2.imwrite("right face.jpg",right_half)

left_face = Image.open("left face.jpg")
right_face = Image.open("right face.jpg")
print("Avg pixel intensity on left face : " ,calcAvgPixel(left_face))
print("Avg pixel intensity on right face : " ,calcAvgPixel(right_face))

r1,g1,b1= calcAvgPixel(left_face)
r2,g2,b2 = calcAvgPixel(right_face)
color1_rgb = sRGBColor(r1, g1, b1);
color2_rgb = sRGBColor(r2, g2, b2);
#dissimilarity = euclideanDist(r1,g1,b1,r2,g2,b2)
#print(dissimilarity)
color1_lab = convert_color(color1_rgb, LabColor);

# Convert from RGB to Lab Color Space
color2_lab = convert_color(color2_rgb, LabColor);

# Find the color difference
delta_e = delta_e_cie2000(color1_lab, color2_lab);

print ("The difference between the 2 color = ", delta_e)

# Impose a threshold either from user or hard-coded
threshold = int(input())
#threshold = 20
if delta_e > threshold:
    print("Asymmetry exists")
cv2.waitKey(0)
cv2.destroyAllWindows()