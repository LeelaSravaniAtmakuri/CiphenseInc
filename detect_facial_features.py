import cv2

face_detect = cv2.CascadeClassifier(r"E:\Anaconda\pkgs\opencv-3.4.1-py36_200\Library\etc\haarcascades\haarcascade_frontalface_default.xml")
eyes_detect = cv2.CascadeClassifier(r"E:\Anaconda\pkgs\opencv-3.4.1-py36_200\Library\etc\haarcascades\haarcascade_eye.xml")
nose_detect = cv2.CascadeClassifier(r"E:\Anaconda\pkgs\opencv-3.4.1-py36_200\Library\etc\haarcascades\nose.xml")
mouth_detect = cv2.CascadeClassifier(r"E:\Anaconda\pkgs\opencv-3.4.1-py36_200\Library\etc\haarcascades\mouth.xml")

cam = cv2.VideoCapture(0)
while True:
    ret, image = cam.read()
    if ret:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = face_detect.detectMultiScale(gray, 1.2, 5)                 # Haarcascades classifier for face
        # print(face)
        for (x, y, w, h) in face:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Draw rectangle around face
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]
            eyes = eyes_detect.detectMultiScale(roi_gray, 1.2)            # Haarcascades classifier for eyes
            nose = nose_detect.detectMultiScale(roi_gray, 1.2, 20)        # Haarcascades classifier for nose
            mouth = mouth_detect.detectMultiScale(roi_gray, 1.2, 50)      # Haarcascades classifier for mouth
            for (ex, ey, ew, eh) in eyes:                                 # Draw rectangle around eyes
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
            for (nx, ny, nw, nh) in nose:                                 # Draw rectangle around nose
                cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 255, 0), 3)
            for (mx, my, mw, mh) in mouth:                                # Draw rectangle around mouth
                cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255, 255), 3)

        cv2.imshow("Detected facial feaures", image)
        if cv2.waitKey(5) & 0xFF == ord('q'):                             # Press q to exit the window
            break
cv2.destroyAllWindows()
