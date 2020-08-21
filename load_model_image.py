import numpy as np
import cv2
import os
import facerecognition as fr



image = cv2.imread(r"C:\\Users\\Chaibedraa\\Downloads\\me.jpeg") # Path for the image you want to test

face_detected, gray = fr.FaceDetection(image)
print("face detected:", face_detected)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'') # Where training file YML is

name = {0: 'Ibrahim'}


for face in face_detected:
	(x, y, w, h) = face
	# roi = reason of intrest 
	roi_gray = gray[y:y+h, x:x+h]
	label, confidence = face_recognizer.predict(roi_gray)
	print("Confidence", confidence)
	print("Label", label)
	fr.DrawShape(image, face)
	predict_name = name[label]
	fr.PutText(image, predict_name, x-12,y-12)



cv2.imshow("Testing Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
