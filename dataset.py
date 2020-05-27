import cv2
import numpy as np
facedetector=cv2.CascadeClassifier('C:/Users/Lenovo/Music/final_year_project/haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
id=int(input('enter the id'))
sample=0
while(1):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facedetector.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+h,y+w),(0,0,255),4)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        sample=sample+1
        cv2.imwrite("C:/Users/Lenovo/Music/dataset/user."+str(id)+'.'+str(sample)+'.jpg',roi_gray)
        cv2.waitKey(100)
    cv2.imshow('img',img)
    cv2.waitKey(1)
    if sample==10:
        break
cam.release()
cv2.destroyAllWindows()
