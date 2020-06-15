import cv2
import numpy as np
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

while True:
	ret, frame = cam.read()

	if ret:
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = model.detectMultiScale(gray_frame,1.3,5)
		for face in faces:
			x,y,w,h = face

			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.imshow("Web Cam Feed", frame)
		key = cv2.waitKey(1)
		if key == ord("q"):
			break
	else:
		break

cam.release()
cv2.destroyAllWindows()

print("End")

# Motion Blur ka reason??