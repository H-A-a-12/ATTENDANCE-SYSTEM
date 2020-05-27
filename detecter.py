import cv2
import csv
import time
id=0 
i=0
d=[]
k=[]
j=[]
#time.sleep(10)
cam=cv2.VideoCapture(0)
facedetector=cv2.CascadeClassifier('C:/Users/Lenovo/Music/final_year_project/haarcascade_frontalface_default.xml')

rec=cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer//trainningData.yml')

#filename='attendance.csv'
while True:
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facedetector.detectMultiScale(gray,1.1,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        print(id)
        print(conf)
        if(id==1):
            i=i+1
            
            d.append('disha')
            k.append('25')
            j.append('yes')
            print(d)
            #writer.writerow({'disha','17','PRESENT'})
                #writer.writerows({'','',''})
          
            cv2.putText(img,'harsh',(150,150),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
        if(id==3):
            i=i+1
            d.append('HARSH')
            k.append('31')
            j.append('yes')
            #writer.writerow({'harsh','31','p'})
            #writer.writerows({'','',''})
               
            cv2.putText(img,'harsh',(150,150),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
        print('record saved')
    cv2.imshow('faces',img)
   # time.sleep(10)
    if cv2.waitKey(1)==13 or i==10:
        break
cam.release()
cv2.destroyAllWindows()
li=['NAME',d[0],'ROLL NO',k[0],'ATTENDANCE',j[0]]
with open('attendance.csv','w',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=li)
    writer.writeheader()
    #name=input('harsh')
    #writer.writerows(a)
    #writer.writerow({'NAME':'harsh','ROLL NO':'31','ATTENDANCE':d[0]})
d=[]
k=[]
j=[]
print(d)
#w.close()




