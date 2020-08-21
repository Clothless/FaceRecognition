import numpy
import os
import cv2

# Face detection function:
def FaceDetection(image):
	# This is to transfer the image from RGB to Gray
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	face_xmlfile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	faces = face_xmlfile.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=3)
	return faces, gray_image


def TrainingData(directory):
	faces = []
	faceID = []

	for path, subdir, filenames in os.walk(directory):
		for filename in filenames:
			if filename.startswith("."):
				print("Skipping this file")
				continue
			ID = os.path.basename(path)
			image_path = os.path.join(path, filename)
			test_image = cv2.imread(image_path)
			if test_image is None:
				print("There is a problem with opening the file")
				continue

			faces_shape, gray_image = FaceDetection(test_image)
			(x, y, w, h) = faces_shape[0]
			roi_gray = gray_image[y:y+w, x:x+h]
			faces.append(roi_gray)
			faceID.append(int(ID))
	return faces, faceID



def TrainClassifier(faces, faceID):
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	face_recognizer.train(faces, numpy.array(faceID))
	return face_recognizer


def DrawShape(test_image, face):
	(x, y, w, h) = face
	cv2.rectangle(test_image, (x, y), (x+w, y+h), (172,172,172), thickness=3)


def PutText(image, label_name, x,y):
	cv2.putText(image, label_name, (x,y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,0,0), 2)

