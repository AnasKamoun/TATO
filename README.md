Facial Recognition Access Control System
This is a Python-based application designed to manage access to secure areas using facial recognition. The system uses OpenCV for face detection, scikit-learn for training an SVM model with HOG features, SQLite for data storage, and Tkinter for a graphical user interface. It supports user management, real-time surveillance, access logging, and notifications for unauthorized access attempts.
Features

User Management: Add, edit, and delete users with associated face images and authorization status.
Real-Time Recognition: Identifies users via webcam feed, logs access attempts, and triggers alerts for unauthorized or unknown individuals.
Access History: View, search, and delete access logs with filtering by user ID or name.
Notifications: Sends email alerts for unauthorized access or impostor attempts (SMTP configuration required).
Model Training: Trains an SVM model using precomputed HOG feature vectors stored in a SQLite database.

Prerequisites

Python 3.8+
Dependencies:
opencv-python
numpy
scikit-learn
scikit-image
joblib
pillow (for Tkinter image handling, if needed)


Hardware:
Webcam for real-time face capture.
Write permissions for creating directories (dataset, imposter) and database (access_control.db).



Installation

Clone the Repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Install Dependencies:
pip install opencv-python numpy scikit-learn scikit-image joblib pillow


Set Up the Database:

The application automatically creates access_control.db on first run.
Ensure the script has write permissions in the project directory.


Configure Notifications (Optional):

Edit the send_email function in access_control_system.py with your SMTP server details (host, port, username, password).
Example:with smtplib.SMTP('smtp.gmail.com', 587) as server:
    server.starttls()
    server.login('your-email@gmail.com', 'your-password')





Usage

Run the Application:
python access_control_system.py


Log In:

Use the default credentials: Username admin, Password admin123.
For production, update the password hashing in the administrateurs table.


Add Users:

Go to the "Users" tab, click "Add User," enter a name, and set authorization status.
Capture 5-10 face images using the webcam (press 'c' to capture, 'q' to quit).


Train the Model:

Ensure at least two users have feature vectors in the database.
Go to the "Train Model" tab and click "Train Model."


Start Surveillance:

In the "Surveillance" tab, click "Start Surveillance" to begin real-time recognition.
The system logs access attempts and displays user names or "Inconnu" (unknown) on the video feed.


Manage History:

In the "History" tab, view access logs with user ID, name, date/time, type, and image path.
Use the search fields to filter by user ID or name.
Select entries and click "Delete Selected" to remove them.



Project Structure
your-repo-name/
├── access_control_system.py  # Main application script
├── dataset/                  # Stores user face images (created automatically)
│   ├── user_1/
│   ├── user_2/
│   └── ...
├── imposter/                 # Stores images of unrecognized attempts
├── access_control.db         # SQLite database (created automatically)
├── face_recognition_model.pkl # Trained SVM model
├── label_dict.pkl            # User ID to name mappings
└── README.md

Database Schema

utilisateurs: Stores user data (id, nom, autorise).
user_features: Stores HOG feature vectors (id, user_id, feature_vector).
acces: Logs access attempts (id, utilisateur_id, date_heure, type, image_path).
notifications: Stores notification settings (id, type, destinataire).
administrateurs: Stores admin credentials (id, username, password_hash).

Troubleshooting

"Inconnu" Label for All Users:
Check console output for Predicted ID and Valid user IDs in database.
Ensure user_id in user_features matches id in utilisateurs.
Retrain the model with high-quality images (good lighting, multiple angles).
Lower confidence_threshold in real_time_recognition (e.g., to 0.2).


Database Errors:
Verify access_control.db exists and is writable.
Run SELECT * FROM utilisateurs; and SELECT user_id FROM user_features; in a SQLite tool to check data.


Camera Issues:
Ensure the webcam is connected and accessible (index 0 in cv2.VideoCapture(0)).



Contributing

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Built with OpenCV, scikit-learn, and Tkinter.
Inspired by access control systems for secure environments.

