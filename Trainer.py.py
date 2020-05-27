import cv2
import numpy as np
import os
from PIL import Image


recognizer=cv2.face.LBPHFaceRecognizer_create()
path='dataset'
def getID(path):
    imagepath=[os.path.join(path,f) for f in os.listdir(path)]
    print(imagepath)
#t=getID(path)
    faces=[]
    IDs=[]
    for i in imagepath:
        faceimg=Image.open(i).convert('L')
        faceNP=np.array(faceimg,np.uint8)
        faces.append(faceNP)
        ID=int(os.path.split(i)[-1].split('.')[1])
        print(ID)
        IDs.append(ID)
        cv2.imshow('trainer',faceNP)
        cv2.waitKey(1)
    return IDs,faces
ids,faces=getID(path)
recognizer.train(faces,np.array(ids))
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()
