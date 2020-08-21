import cv2
import sys

count = 0

video_stream = cv2.VideoCapture(0)

while True:
	ret, frame = video_stream.read()
	cv2.imshow("This is the test frame", frame)

	cv2.imwrite(r"C:/Users/Chaibedraa/ML/facerecognition/images/0/image%04i.jpeg" %count, frame)
	count += 1

	if cv2.waitKey(10) == ord('q'):
		break
