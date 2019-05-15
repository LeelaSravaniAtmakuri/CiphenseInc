import cv2
from imutils import face_utils
import numpy as np
import dlib
import imutils

face_detect = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\ADMIN\Downloads\shape_predictor_68_face_landmarks.dat")

image = cv2.imread(r"D:\download4.jpg")
# cv2.imshow("im",imag)
image = cv2.resize(image, (500, 500))
grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = face_detect(grayimage, 1)

for (i, rect) in enumerate(rects):
    shape = predictor(grayimage, rect)
    shape = face_utils.shape_to_np(shape)
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(image, "Face {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HEYSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for (x, y) in shape:
        cv2.line(image, (x, y), (x + 0.01, y + 0.01), (0, 255, 0), 1)

cv2.imshow("Image with facial marks detected", image)
cv2.waitKey(0)
cv2.destroyAllWindows()