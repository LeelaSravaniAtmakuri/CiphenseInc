import cv2
from PIL import Image
import numpy as np

face_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_frontalface_default.xml")
eyes_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_eye.xml")
nose_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\nose.xml")
mouth_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\mouth.xml")

image = cv2.imread(r"D:\Intern\test images\test11.jpg")
#image = imutils.resize(image,width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert color image to gray image

def calcAvgPixel(im):
    pixels = list(im.getdata())
    #print(pixels)
    pixel_list = [x for sets in pixels for x in sets]
    #print(pixel_list)
    pixel_red = pixel_list[::3]
    pixel_green = pixel_list[1::3]
    pixel_blue = pixel_list[2::3]
    red = int(np.mean(np.array(pixel_red)))
    green = int(np.mean(np.array(pixel_green)))
    blue = int(np.mean(np.array(pixel_blue)))
    return (red, green, blue);

face = face_detect.detectMultiScale(gray, 1.2, 5)

if len(face)!=0:
    image1 = image.copy()
    for (x, y, w, h) in face:  # Loop over the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y+h,x:x+w]
        roi_color_f = image1[y:y + h, x:x + w]
        cv2.imwrite("face.jpg", roi_color_f)
        im_f = Image.open(r"face.jpg")
        print("Avg pixel intensity for face:",calcAvgPixel(im_f))

        eyes = eyes_detect.detectMultiScale(roi_gray, 1.2)  # Haarcascades classifier for eyes
        i=1
        if len(eyes)!=0:
            for (ex, ey, ew, eh) in eyes:  # Draw rectangle around eyes
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                roi_color_e = roi_color_f[ey:ey+eh,ex:ex+ew]
                cv2.imwrite("eyes.jpg", roi_color_e)
                im_e = Image.open(r"eyes.jpg")
                print("Avg pixel intensity for eye",i,":", calcAvgPixel(im_e))
                i=i+1
                if i>2:
                    break
        nose = nose_detect.detectMultiScale(roi_gray,1.1)  # Haarcascades classifier for nose
        j=1
        if len(nose)!=0:
            for (nx, ny, nw, nh) in nose:  # Draw rectangle around nose
                cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 255, 0), 2)
                roi_color_n = roi_color_f[ny:ny + nh, nx:nx + nw]
                cv2.imwrite("nose.jpg", roi_color_n)
                im_n = Image.open(r"nose.jpg")
                print("Avg pixel intensity for nose:", calcAvgPixel(im_n))
                j=j+1
                if j>1:
                    break

        mouth = mouth_detect.detectMultiScale(roi_gray, 1.2,45)  # Haarcascades classifier for mouth
        k=1
        if len(mouth)!=0:
            for (mx, my, mw, mh) in mouth:  # Draw rectangle around mouth
                cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255, 255), 2)
                roi_color_m = roi_color_f[my:my + mh, mx:mx + mw]
                cv2.imwrite("mouth.jpg", roi_color_m)
                im_m = Image.open(r"mouth.jpg")
                print("Avg pixel intensity for mouth:",calcAvgPixel(im_m))
                k=k+1
                if k>1:
                    break
cv2.imshow("Detected facial feaures", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
