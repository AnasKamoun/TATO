# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 14:10:56 2025

@author: PC
"""
import os
import cv2
import numpy as np
import joblib  # Pour sauvegarder et charger le modèle
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def load_images_and_labels(data_dir):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    labels = []
    features = []
    label_dict = {}
    label_count = 0

    if not os.path.isdir(data_dir):
        print(f"Erreur: Le répertoire '{data_dir}' n'existe pas.")
        return np.array([]), np.array([]), {}

    for subject in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject)
        if os.path.isdir(subject_path):
            label_dict[label_count] = subject
            for image_name in os.listdir(subject_path):
                if not image_name.lower().endswith(valid_extensions):
                    continue
                image_path = os.path.join(subject_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                face, _ = detect_face(image)
                if face is not None:
                    face = preprocess(face)
                    hog_features = extract_hog_features(face)
                    features.append(hog_features)
                    labels.append(label_count)
            label_count += 1

    return np.array(features), np.array(labels), label_dict

def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return image[y:y+h, x:x+w], (x, y, w, h)
    return None, None

def preprocess(face):
    face = cv2.resize(face, (64, 64))
    face = cv2.equalizeHist(face)
    return face / 255.0

def extract_hog_features(image):
    return hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

if __name__ == "__main__":
    data_dir = r"./dataset"
    data, labels, label_dict = load_images_and_labels(data_dir)
    
    if data.size == 0 or labels.size == 0:
        print("Aucune donnée chargée. Vérifiez votre dataset.")
        exit(1)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    pipeline = make_pipeline(StandardScaler(), SVC(probability=True))
    param_grid = {'svc__C': [0.1, 1, 10, 100], 'svc__kernel': ['linear', 'rbf'], 'svc__gamma': ['scale', 'auto']}
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f"Meilleurs paramètres : {grid.best_params_}")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision du modèle sur test: {accuracy * 100:.2f}%")
    joblib.dump(X_test,"X_test.pkl")
    joblib.dump(y_test,"y_test.pkl")
    joblib.dump(best_model, "face_recognition_model.pkl")
    joblib.dump(label_dict, "label_dict.pkl")
    print("Modèle, labels, et variables sauvegardés avec succès.")