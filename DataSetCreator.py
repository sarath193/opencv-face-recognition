import cv2
import os

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

identity = input('Enter name: ')
counter = 0
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        counter += 1
        path = os.path.join(os.getcwd(), "dataSet", "User_" +
                            str(identity) + "_" + str(counter) + ".jpg")

        cv2.imwrite(path, gray[y:y + h, x:x + w])
        print ("Written at ", path)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.waitKey(100)

    cv2.imshow("Face Detection Window", img)

    cv2.waitKey(1)
    if(counter >= 30):
        break

cap.release()
cv2.destroyAllWindows()
