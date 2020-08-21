import numpy as np
import cv2
import os
import facerecognition as fr



image = cv2.imread(r"C:\\Users\\Chaibedraa\\Downloads\\me.jpeg")

face_detected, gray = fr.FaceDetection(image)
print("face detected:", face_detected)


faces, faceID = fr.TrainingData(r'C:\\Users\\Chaibedraa\\ML\\facerecognition\\images\\')
face_recognizer = fr.TrainClassifier(faces, faceID)
face_recognizer.save(r'C:\\Users\\Chaibedraa\\ML\\facerecognition\\trainingData.yml') # It will save the trained model


name = {0: "Ibrahim", 1: "Aaron_Eckhart",}

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

