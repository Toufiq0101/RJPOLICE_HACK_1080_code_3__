import cv2
import os

new_path = 'I:/ProjectX/face'

def save(img, name, bbox, width=180, height=227):
    x, y, w, h = bbox
    imgCrop = img[y:h, x:w]
    imgCrop = cv2.resize(imgCrop, (width, height))
    cv2.imwrite(name + ".jpg", imgCrop)

def face_detection_in_video(video_path):
  
    cap = cv2.VideoCapture(video_path)

  
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

   
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

      
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for counter, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (220, 255, 220), 1)
            save(frame, os.path.join(new_path, str(counter)), (x, y, x + w, y + h))

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

video_path = 'I:/ProjectX/face/video1.mp4' 
face_detection_in_video(video_path)
