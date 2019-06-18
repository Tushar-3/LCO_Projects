import numpy as np
import cv2 as cv

cam=cv.VideoCapture(0)
# This cascades are imported from https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
detector_face=cv.CascadeClassifier("haar_frontal.xml")
detector_eye=cv.CascadeClassifier("haarcascade_eye.xml")

while True:
    _ ,frame=cam.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces=detector_face.detectMultiScale(gray,1.7,5)

    for x,y,w,h in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        eyes = detector_eye.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
    cv.imshow("face eye Dect",frame)
    key=cv.waitKey(1)

    if key & 0xff == ord('q'):
        cv.destroyAllWindows()
        break
cam.release()
