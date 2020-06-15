import numpy as np
import os
import cv2

dataset_path= "./data/"

face_data = []
labels = []

for fx in os.listdir(dataset_path):
	if fx.endswith(".npy"):
		l = fx.split(".")[0]
		face_item = np.load(dataset_path + fx)
		print("loaded : ",l)
		face_data.append(face_item)
		for i in range(len(face_item)):
			labels.append(l)
labels = np.array(labels)
face_data = np.concatenate(face_data, axis=0)

print(face_data.shape)
print(labels.shape)

#kNN Code from here

def distance(pA,pB):
    return np.sum((pB-pA)**2)**0.5

def kNN (X, Y, x_query, k=5):
	m = X.shape[0]
	distances = []
	for i in range(m):
		dis = distance(x_query, X[i])
		distances.append((dis,Y[i]))

	distances = sorted(distances)
	distances = distances[:k]
	results = np.array(distances)[:,1]
	uniq_labels, counts = np.unique(results, return_counts = True)
	ans = uniq_labels[counts.argmax()]
	return ans

#Test Face Recog
cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
while True:
    ret, frame = cam.read()
    if ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
        for face in faces:
            x,y,w,h = face
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            offset = 10
            face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
            face_section = cv2.resize(face_section,(100,100))
            name = kNN(face_data,labels,face_section.reshape(1,-1))
            cv2.putText(frame,name.title(),(x,y-10),cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Web Cam Feed", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    else:
        break

cam.release()
cv2.destroyAllWindows()