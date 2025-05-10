import os
import cv2
import numpy as np
import joblib
import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
import winsound
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import io
import shutil

# Utility Functions
def array_to_blob(array):
    out = io.BytesIO()
    np.save(out, array)
    out.seek(0)
    return out.read()

def blob_to_array(blob):
    out = io.BytesIO(blob)
    return np.load(out)

# Database Initialization
def init_db():
    conn = sqlite3.connect('access_control.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS utilisateurs
                 (id INTEGER PRIMARY KEY, nom TEXT, autorise BOOLEAN)''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_features
                 (id INTEGER PRIMARY KEY, user_id INTEGER, feature_vector BLOB,
                  FOREIGN KEY (user_id) REFERENCES utilisateurs(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS acces
                 (id INTEGER PRIMARY KEY, utilisateur_id INTEGER, date_heure DATETIME, type TEXT, image_path TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS notifications
                 (id INTEGER PRIMARY KEY, type TEXT, destinataire TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS administrateurs
                 (id INTEGER PRIMARY KEY, username TEXT, password_hash TEXT)''')
    c.execute("INSERT OR IGNORE INTO administrateurs (username, password_hash) VALUES (?, ?)", 
              ("admin", "admin123"))  # Use proper hashing in production
    c.execute("INSERT OR IGNORE INTO notifications (type, destinataire) VALUES (?, ?)", 
              ("email", "admin@example.com"))
    conn.commit()
    conn.close()

# Facial Recognition Functions
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

def train_model():
    conn = sqlite3.connect('access_control.db')
    c = conn.cursor()
    c.execute("SELECT user_id, feature_vector FROM user_features")
    data = c.fetchall()
    if not data:
        print("No data to train.")
        return None, None
    labels = [row[0] for row in data]
    features = [blob_to_array(row[1]) for row in data]
    c.execute("SELECT id, nom FROM utilisateurs")
    label_dict = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    if len(set(labels)) < 2:
        print("Need at least two distinct users to train the model.")
        return None, None
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    pipeline = make_pipeline(StandardScaler(), SVC(probability=True))
    param_grid = {'svc__C': [1, 10], 'svc__kernel': ['rbf'], 'svc__gamma': ['scale']}
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    joblib.dump(best_model, "face_recognition_model.pkl")
    joblib.dump(label_dict, "label_dict.pkl")
    print("Model trained and saved.")
    return best_model, label_dict

# Notification Functions
def send_email(recipient, message):
    msg = MIMEText(message)
    msg['Subject'] = 'Access Control Alert'
    msg['From'] = 'sender@example.com'
    msg['To'] = recipient
    try:
        with smtplib.SMTP('smtp.example.com') as server:  # Configure with real SMTP server
            server.login('username', 'password')
            server.send_message(msg)
    except Exception as e:
        print(f"Email sending failed: {e}")

def send_notification(type_acces, user_id=None, image_path=None):
    conn = sqlite3.connect('access_control.db')
    c = conn.cursor()
    c.execute("SELECT type, destinataire FROM notifications")
    recipients = c.fetchall()
    conn.close()
    message = f"{type_acces.capitalize()} detected"
    if user_id:
        message += f" by user ID {user_id}"
    if image_path:
        message += f". Image saved at {image_path}"
    for n_type, dest in recipients:
        if n_type == "email":
            send_email(dest, message)

# Real-Time Recognition
def real_time_recognition(root, model, label_dict):
    if model is None or label_dict is None:
        messagebox.showerror("Error", "Model not trained. Please train the model first.")
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera.")
        return
    confidence_threshold = 0.4  # Lowered for better recognition

    def process_frame():
        conn = sqlite3.connect('access_control.db')
        c = conn.cursor()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame.")
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face, coords = detect_face(gray)
                if face is not None and coords is not None:
                    face_proc = preprocess(face)
                    feature = extract_hog_features(face_proc).reshape(1, -1)
                    prediction = model.predict(feature)
                    confidence = model.predict_proba(feature).max()
                    print(f"Confidence: {confidence:.2f}, Predicted ID: {prediction[0]}")  # Debug
                    try:
                        if confidence > confidence_threshold:
                            user_id = int(prediction[0])
                            c.execute("SELECT nom, autorise FROM utilisateurs WHERE id= CAST(? AS INTEGER)", (user_id,))
                            result = c.fetchone()
                            print(result)
                            if result:
                                nom, autorise = result
                                label_text = nom
                                color = (0, 255, 0) if autorise else (0, 0, 255)
                                access_type = "autorise" if autorise else "non_autorise"
                                c.execute("INSERT INTO acces (utilisateur_id, date_heure, type) VALUES (?, ?, ?)",
                                          (user_id, datetime.now(), access_type))
                                conn.commit()
                                print(f"Access logged for user {nom} ({access_type})")
                                if not autorise:
                                    send_notification("non_autorise", user_id)
                            else:
                                label_text = "azzzz"
                                color = (0, 0, 255)
                        else:
                            label_text = "Inconnu"
                            color = (0, 0, 255)
                            image_path = f"imposter/intrusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            cv2.imwrite(image_path, frame)
                            c.execute("INSERT INTO acces (date_heure, type, image_path) VALUES (?, ?, ?)",
                                      (datetime.now(), "imposteur", image_path))
                            conn.commit()
                            send_notification("imposteur", image_path=image_path)
                            winsound.Beep(1000, 500)
                    except sqlite3.Error as e:
                        print(f"Database error: {e}")
                        label_text = "Erreur"
                        color = (0, 0, 255)
                    x, y, w, h = coords
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{label_text} ({confidence:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imshow('Surveillance', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            conn.close()
            cap.release()
            cv2.destroyAllWindows()

    threading.Thread(target=process_frame, daemon=True).start()

# GUI Application
class AccessControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Access Control System")
        self.model = joblib.load("face_recognition_model.pkl") if os.path.exists("face_recognition_model.pkl") else None
        self.label_dict = joblib.load("label_dict.pkl") if os.path.exists("label_dict.pkl") else None
        self.login_screen()

    def login_screen(self):
        self.clear_frame()
        tk.Label(self.root, text="Admin Login").pack()
        tk.Label(self.root, text="Username:").pack()
        username = tk.Entry(self.root)
        username.pack()
        tk.Label(self.root, text="Password:").pack()
        password = tk.Entry(self.root, show="*")
        password.pack()
        tk.Button(self.root, text="Login", command=lambda: self.check_login(username.get(), password.get())).pack()

    def check_login(self, username, password):
        conn = sqlite3.connect('access_control.db')
        c = conn.cursor()
        c.execute("SELECT password_hash FROM administrateurs WHERE username=?", (username,))
        result = c.fetchone()
        conn.close()
        if result and result[0] == password:  # Simple check; use hashing in production
            self.main_interface()
        else:
            messagebox.showerror("Error", "Invalid credentials")

    def clear_frame(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def main_interface(self):
        self.clear_frame()
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        # Surveillance Tab
        surv_frame = ttk.Frame(notebook)
        notebook.add(surv_frame, text="Surveillance")
        tk.Button(surv_frame, text="Start Surveillance",
                  command=lambda: real_time_recognition(self.root, self.model, self.label_dict)).pack()

        # User Management Tab
        user_frame = ttk.Frame(notebook)
        notebook.add(user_frame, text="Users")
        self.user_tree = ttk.Treeview(user_frame, columns=("ID", "Name", "Authorized"), show="headings")
        self.user_tree.heading("ID", text="ID")
        self.user_tree.heading("Name", text="Name")
        self.user_tree.heading("Authorized", text="Authorized")
        self.user_tree.pack()
        tk.Button(user_frame, text="Add User", command=self.add_user_window).pack()
        tk.Button(user_frame, text="Edit User", command=self.edit_user).pack()
        tk.Button(user_frame, text="Delete User", command=self.delete_user).pack()
        self.load_users()

        # History Tab
        hist_frame = ttk.Frame(notebook)
        notebook.add(hist_frame, text="History")
        self.hist_tree = ttk.Treeview(hist_frame, columns=("ID", "UserID", "DateTime", "Type", "Image"), show="headings")
        self.hist_tree.heading("ID", text="ID")
        self.hist_tree.heading("UserID", text="User ID")
        self.hist_tree.heading("DateTime", text="Date/Time")
        self.hist_tree.heading("Type", text="Type")
        self.hist_tree.heading("Image", text="Image Path")
        self.hist_tree.pack()
        self.load_history()

        # Train Model Tab
        train_frame = ttk.Frame(notebook)
        notebook.add(train_frame, text="Train Model")
        tk.Button(train_frame, text="Train Model", command=self.train_model_gui).pack()

    def load_users(self):
        for item in self.user_tree.get_children():
            self.user_tree.delete(item)
        conn = sqlite3.connect('access_control.db')
        c = conn.cursor()
        c.execute("SELECT id, nom, autorise FROM utilisateurs")
        for row in c.fetchall():
            self.user_tree.insert("", "end", values=row)
        conn.close()

    def load_history(self):
        for item in self.hist_tree.get_children():
            self.hist_tree.delete(item)
        conn = sqlite3.connect('access_control.db')
        c = conn.cursor()
        c.execute("SELECT id, utilisateur_id, date_heure, type, image_path FROM acces")
        for row in c.fetchall():
            self.hist_tree.insert("", "end", values=row)
        conn.close()

    def add_user_window(self):
        win = tk.Toplevel(self.root)
        win.title("Add User")
        tk.Label(win, text="Name:").pack()
        name_entry = tk.Entry(win)
        name_entry.pack()
        auth_var = tk.BooleanVar(value=True)
        tk.Checkbutton(win, text="Authorized", variable=auth_var).pack()
        tk.Button(win, text="Save", command=lambda: self.save_user_and_capture(name_entry.get(), auth_var.get(), win)).pack()

    def save_user_and_capture(self, name, authorized, window):
        if not name:
            messagebox.showerror("Error", "Name is required")
            return
        conn = sqlite3.connect('access_control.db')
        c = conn.cursor()
        c.execute("INSERT INTO utilisateurs (nom, autorise) VALUES (?, ?)", (name, authorized))
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        self.load_users()
        window.destroy()
        self.capture_user_images(user_id)

    def capture_user_images(self, user_id):
        win = tk.Toplevel(self.root)
        win.title(f"Capture Images for User {user_id}")
        tk.Label(win, text="Press 'c' to capture a face image, 'q' to quit").pack()
        cap = cv2.VideoCapture(0)
        count = 0
        os.makedirs(f"dataset/user_{user_id}", exist_ok=True)
        conn = sqlite3.connect('access_control.db')
        c = conn.cursor()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow('Capture', frame)
                key = cv2.waitKey(1)
                if key == ord('c'):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face, coords = detect_face(gray)
                    if face is not None:
                        image_path = f"dataset/user_{user_id}/face_{count}.jpg"
                        cv2.imwrite(image_path, face)
                        face_proc = preprocess(face)
                        feature = extract_hog_features(face_proc)
                        feature_blob = array_to_blob(feature)
                        c.execute("INSERT INTO user_features (user_id, feature_vector) VALUES (?, ?)", (user_id, feature_blob))
                        conn.commit()
                        count += 1
                elif key == ord('q'):
                    break
        finally:
            conn.close()
            cap.release()
            cv2.destroyAllWindows()
            win.destroy()

    def edit_user(self):
        selected = self.user_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Select a user to edit")
            return
        user_id = self.user_tree.item(selected, "values")[0]
        conn = sqlite3.connect('access_control.db')
        c = conn.cursor()
        c.execute("SELECT nom, autorise FROM utilisateurs WHERE id=?", (user_id,))
        name, auth = c.fetchone()
        conn.close()
        win = tk.Toplevel(self.root)
        win.title("Edit User")
        tk.Label(win, text="Name:").pack()
        name_entry = tk.Entry(win)
        name_entry.insert(0, name)
        name_entry.pack()
        auth_var = tk.BooleanVar(value=auth)
        tk.Checkbutton(win, text="Authorized", variable=auth_var).pack()
        tk.Button(win, text="Update Images", command=lambda: self.update_user_images(user_id)).pack()
        tk.Button(win, text="Save", command=lambda: self.update_user(user_id, name_entry.get(), auth_var.get(), win)).pack()

    def update_user_images(self, user_id):
        conn = sqlite3.connect('access_control.db')
        c = conn.cursor()
        c.execute("DELETE FROM user_features WHERE user_id=?", (user_id,))
        conn.commit()
        conn.close()
        user_dir = f"dataset/user_{user_id}"
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)
        self.capture_user_images(user_id)

    def update_user(self, user_id, name, authorized, window):
        conn = sqlite3.connect('access_control.db')
        c = conn.cursor()
        c.execute("UPDATE utilisateurs SET nom=?, autorise=? WHERE id=?", (name, authorized, user_id))
        conn.commit()
        conn.close()
        self.load_users()
        window.destroy()

    def delete_user(self):
        selected = self.user_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Select a user to delete")
            return
        user_id = self.user_tree.item(selected, "values")[0]
        conn = sqlite3.connect('access_control.db')
        c = conn.cursor()
        c.execute("DELETE FROM utilisateurs WHERE id=?", (user_id,))
        c.execute("DELETE FROM user_features WHERE user_id=?", (user_id,))
        conn.commit()
        conn.close()
        user_dir = f"dataset/user_{user_id}"
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)
        self.load_users()

    def train_model_gui(self):
        conn = sqlite3.connect('access_control.db')
        c = conn.cursor()
        c.execute("SELECT COUNT(DISTINCT user_id) FROM user_features")
        count = c.fetchone()[0]
        conn.close()
        if count < 2:
            messagebox.showerror("Error", "Need at least two users with features to train the model.")
        else:
            try:
                self.model, self.label_dict = train_model()
                if self.model is None:
                    messagebox.showerror("Error", "Training failed. Ensure at least two users have feature data.")
                else:
                    messagebox.showinfo("Info", "Model training completed")
            except Exception as e:
                messagebox.showerror("Error", f"Training failed: {str(e)}")

if __name__ == "__main__":
    init_db()
    root = tk.Tk()
    app = AccessControlApp(root)
    root.mainloop()