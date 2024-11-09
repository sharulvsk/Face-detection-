import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

def train_face_recognizer(image_folder):
    faces = []
    labels = []
    label_dict = {}
    label_counter = 0
    
    for person_name in os.listdir(image_folder):
        person_folder = os.path.join(image_folder, person_name)
        if os.path.isdir(person_folder):
            if person_name not in label_dict:
                label_dict[person_name] = label_counter
                label_counter += 1

            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img)
                    labels.append(label_dict[person_name])
    
    recognizer.train(faces, np.array(labels))
    recognizer.save('face_trainer.yml')
    print("Training complete. Model saved as 'face_trainer.yml'.")
    return label_dict

label_dict = train_face_recognizer(r'C:\From Destop\Dataset_face')

recognizer.read('face_trainer.yml')

image_path = r'C:\From Destop\shankaripriya s profile.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    label_dict = train_face_recognizer(r'C:\From Destop\Dataset_face')

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        
        label, confidence = recognizer.predict(roi_gray)
        
        name = list(label_dict.keys())[list(label_dict.values()).index(label)]
        if confidence < 100:
            cv2.putText(image, f"Person: {name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            cv2.putText(image, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Face Recognition', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
