import numpy as np
import cv2
import os
import facerecognition as fr


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'') # Where training file YML is


cap = cv2.VideoCapture(0) # 0 is the camera, if you want to test it on a video just put the video path
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

name = {0: 'Ibrahim'}

while True:
	ret, video = cap.read()
	face_detected, gray_video = fr.FaceDetection(video)
	print("Face Detected", face_detected)
	for (x, y, h, w) in face_detected:
		cv2.rectangle(video, (x,y), (x+w, y+h), (172,172,172), thikness=5)




for face in face_detected:
	(x, y, w, h) = face
	# roi = reason of intrest 
	roi_gray = gray_video[y:y+h, x:x+h]
	label, confidence = face_recognizer.predict(roi_gray)
	print("Confidence", confidence)
	print("Label", label)
	fr.DrawShape(video, face)
	predict_name = name[label]
	fr.PutText(video, predict_name, x-12,y-12)


cv2.imshow("Face Detected", video)

if cv2.waitKey(10) == ord('q'):
	break