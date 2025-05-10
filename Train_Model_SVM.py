# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 14:10:56 2025

@author: PC
"""
import os
import cv2
import numpy as np
import joblib
import sqlite3
import io
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def blob_to_array(blob):
    out = io.BytesIO(blob)
    return np.load(out)


def load_images_and_labels():
    # Connect to the SQLite database
    conn = sqlite3.connect('access_control.db')
    c = conn.cursor()

    # Fetch feature vectors and user IDs from user_features
    c.execute("SELECT user_id, feature_vector FROM user_features")
    data = c.fetchall()

    if not data:
        print("No feature vectors found in the database.")
        conn.close()
        return np.array([]), np.array([]), {}

    # Extract features and labels
    features = []
    labels = []
    for user_id, feature_blob in data:
        feature = blob_to_array(feature_blob)
        features.append(feature)
        labels.append(user_id)

    # Fetch user names for label dictionary
    c.execute("SELECT id, nom FROM utilisateurs")
    label_dict = {row[0]: row[1] for row in c.fetchall()}

    conn.close()

    # Convert to numpy arrays
    return np.array(features), np.array(labels), label_dict


def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return image[y:y + h, x:x + w], (x, y, w, h)
    return None, None


def preprocess(face):
    face = cv2.resize(face, (64, 64))
    face = cv2.equalizeHist(face)
    return face / 255.0


def extract_hog_features(image):
    return hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)


if __name__ == "__main__":
    # Load data from the database
    data, labels, label_dict = load_images_and_labels()

    if data.size == 0 or labels.size == 0:
        print("Aucune donnée chargée. Vérifiez votre base de données.")
        exit(1)

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Train the model
    pipeline = make_pipeline(StandardScaler(), SVC(probability=True))
    param_grid = {'svc__C': [0.1, 1, 10, 100], 'svc__kernel': ['linear', 'rbf'], 'svc__gamma': ['scale', 'auto']}
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)
    grid.fit(X_train, y_train)

    # Evaluate the model
    best_model = grid.best_estimator_
    print(f"Meilleurs paramètres : {grid.best_params_}")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision du modèle sur test: {accuracy * 100:.2f}%")

    # Save the model and related data
    joblib.dump(X_test, "X_test.pkl")
    joblib.dump(y_test, "y_test.pkl")
    joblib.dump(best_model, "face_recognition_model.pkl")
    joblib.dump(label_dict, "label_dict.pkl")
    print("Modèle, labels, et variables sauvegardés avec succès.")