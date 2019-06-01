from collections import OrderedDict
import numpy as np
import cv2
import dlib
import imutils

LIPS_IDXS = OrderedDict([("lips", (48,49,50,51,52,53,54,55,56,57,58,59,60))])

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"F:\shape_predictor_68_face_landmarks.dat")

img = cv2.imread(r'F:\test.jpg')
img = imutils.resize(img, width=600)
cv2.imshow("Original",img)
overlay = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img)
detections = detector(gray, 0)
for k,d in enumerate(detections):
    shape = predictor(gray, d)
    for (_, name) in enumerate(LIPS_IDXS.keys()):
        pts = np.zeros((len(LIPS_IDXS[name]), 2), np.int32)
        for i,j in enumerate(LIPS_IDXS[name]):
            pts[i] = [shape.part(j).x, shape.part(j).y]

        pts = pts.reshape((-1,1,2))
        cv2.polylines(overlay,[pts],True,(0,255,0),thickness = 1)
        #img_crop = overlay[d.top():d.bottom(),d.left():d.right()]
    cv2.imshow("Image", overlay)
    #cv2.imshow('Cropped',img_crop)
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()