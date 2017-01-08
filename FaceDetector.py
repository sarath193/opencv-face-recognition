import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
recognizer = cv2.face.createLBPHFaceRecognizer()

recognizer.load("recognizer\\trainingData.yml")
ID = 0

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h),(0,0,255), 2)
        ID, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(ID==1):
            ID = "Sarath"
        if(ID==2):
            ID = "Obama"
        cv2.putText(img, str(ID), (x+w,int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 2,510)

        
    cv2.imshow("Face Detection Window", img)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
