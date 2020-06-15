import cv2
import numpy as np
cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
face_data = []
dataset_path = "./data/"
name = input("Enter your Name : ")
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
            face_data.append(face_section)
            cv2.imshow("Cropped Face", face_section)
        cv2.imshow("Web Cam Feed", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    else:
        break

cam.release()
cv2.destroyAllWindows()

face_data = np.array(face_data)
face_data = face_data.reshape(face_data.shape[0],-1)
print(face_data.shape)

np.save(dataset_path +name+".npy",face_data)
print("Data saved")