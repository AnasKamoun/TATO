# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 14:27:07 2025

@author: PC
"""
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Chargement du modèle et des données de test
model = joblib.load("face_recognition_model.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

# Prédictions et probabilités
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilité de la classe positive

# Calcul des métriques
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title("Matrice de Confusion")
plt.show()

# Rapport de classification
print("Rapport de classification:\n", classification_report(y_test, y_pred))

# # Courbe ROC
# fpr, tpr, _ = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
# plt.xlabel("Taux de faux positifs")
# plt.ylabel("Taux de vrais positifs")
# plt.title("Courbe ROC")
# plt.legend(loc="lower right")
# plt.show()

# Affichage des métriques
print(f"Exactitude: {accuracy:.2f}")
print(f"Précision: {precision:.2f}")
print(f"Rappel: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
