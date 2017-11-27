""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:/Users/hyoung/Documents/_Softdes/Toolbox/Toolbox-ComputerVision/facefile.xml')
kernel = np.ones((21, 21), 'uint8')
while(True):

    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20, 20))
    for (x, y, w, h) in faces:
        frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h, x:x+w, :], kernel)
        cv2.circle(frame,(int(x+h/2),int(y+x/2)), int(h/2), (0,0,255), -1)
        cv2.circle(frame,(int(x+h/4),int(y+x/4)), 10, (0,0,0), -1)
        cv2.circle(frame,(int(x+3*h/4),int(y+x/4)), 10, (0,0,0), -1)
        cv2.circle(frame,(int(x+2*h/4),int(y+3*x/4)), 30, (0,0,0), -1)


    # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()