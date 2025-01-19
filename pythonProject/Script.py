import os
import logging
import time
import threading
import pyodbc
import cv2
import face_recognition
from sklearn.neighbors import NearestNeighbors
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Configure the logging
logging.basicConfig(level=logging.INFO)

# Set up Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})

# Path to the folder where the images are stored
IMAGE_FOLDER = r"C:\Users\dell\source\repos\PatientSystem\images"  # Replace with your actual path
UPLOAD_FOLDER = r"C:\Users\dell\source\repos\PatientSystem\uploads"  # Path for uploaded files
base_url = "http://localhost:5000"  # Replace with your actual base URL

# Flag to ensure initialization happens only once
encodings_initialized = False
encodings_lock = threading.Lock()

# Initialize the SimpleFacerec class
class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.patient_data = {}
        self.knn = None
        self.lock = threading.Lock()

    def connect_to_database(self):
        try:
            connection = pyodbc.connect(
                'DRIVER={SQL Server};'
                'SERVER=DESKTOP-CLQGA5Q\\SQLEXPRESS;'  # Replace with your server
                'DATABASE=PatientSystemDB;'  # Replace with your database name
                'Trusted_Connection=yes;'
            )
            return connection
        except Exception as e:
            logging.error(f"Database connection error: {e}")
            return None

    def load_encoding_images(self):
        logging.info("Loading face encodings...")
        start_time = time.time()

        connection = self.connect_to_database()
        if connection is None:
            return

        try:
            cursor = connection.cursor()
            cursor.execute("SELECT Id, Name, Dob, Mobileno, Nationalno, FaceImg FROM dbo.Patients")
            patients = cursor.fetchall()

            if not patients:
                logging.warning("No patient data found in the database.")
                return

            encodings = []
            names = []
            patient_metadata = {}

            for patient in patients:
                patient_id, name, dob, mobile_no, national_no, face_img = patient
                image_file = os.path.join(IMAGE_FOLDER, str(face_img))
                logging.info(f"Processing image: {image_file}")

                if not os.path.exists(image_file):
                    logging.warning(f"Image file {face_img} not found at path: {image_file}")
                    continue

                img = cv2.imread(image_file)
                img = cv2.resize(img, (320, 320))
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                encodings_in_image = face_recognition.face_encodings(rgb_img)
                if not encodings_in_image:
                    logging.warning(f"No faces found in image {face_img}")
                    continue

                encodings.append(encodings_in_image[0])
                names.append(name)

                face_img_url = f"{base_url}/images/{os.path.basename(image_file)}"
                patient_metadata[name] = {
                    "mobileno": mobile_no,
                    "nationalno": national_no,
                    "id": patient_id,
                    "dob": dob,
                    "faceImg": face_img_url
                }

            if encodings:
                self.knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
                self.knn.fit(encodings)

                self.known_face_encodings = encodings
                self.known_face_names = names
                self.patient_data = patient_metadata

                logging.info(f"Loaded {len(self.known_face_encodings)} face encodings.")
            else:
                logging.warning("No face encodings were loaded.")

        except Exception as e:
            logging.error(f"Error loading encodings: {e}")
        finally:
            connection.close()

    def compare_faces(self, unknown_image, tolerance=0.5):
        small_image = cv2.resize(unknown_image, (320, 240))
        unknown_encoding = face_recognition.face_encodings(small_image)
        if not unknown_encoding:
            logging.warning("No face found in the image.")
            return None

        distances, indices = self.knn.kneighbors([unknown_encoding[0]], n_neighbors=1)
        if distances[0][0] < tolerance:
            return self.known_face_names[indices[0][0]]
        return None


# Flask route for serving images
@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory(IMAGE_FOLDER, filename)


# Route to detect face and find matching patient
@app.route('/detectAndFind', methods=['POST'])
def detect_and_find():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    unknown_image = face_recognition.load_image_file(file_path)

    if not facerec.knn:
        logging.error("KNN model is not initialized or no face encodings are loaded.")
        return jsonify({"error": "Face encodings are not available."}), 500

    tolerance = 0.5
    matched_name = facerec.compare_faces(unknown_image, tolerance)
    os.remove(file_path)

    if matched_name:
        patient_data = facerec.patient_data.get(matched_name)
        return jsonify({"isMatch": True, "patientName": matched_name, "patientData": patient_data})

    return jsonify({"isMatch": False, "patientName": "Unknown", "patientData": {}})


# Route to reload encodings (e.g., after a database update)
@app.route('/reload_encodings', methods=['GET'])
def reload_encodings():
    try:
        facerec.known_face_encodings.clear()
        facerec.known_face_names.clear()
        facerec.patient_data.clear()

        facerec.load_encoding_images()
        logging.info("Encodings reloaded successfully.")
        return jsonify({"message": "Encodings reloaded successfully."}), 200

    except Exception as e:
        logging.error(f"Error during reload: {e}")
        return jsonify({"error": "Failed to reload encodings.", "details": str(e)}), 500


# Initialize the SimpleFacerec instance
facerec = SimpleFacerec()

# Function to load encodings after the app restarts
def load_encodings_on_restart():
    logging.info("Flask server restarted, loading face encodings...")
    facerec.load_encoding_images()
    logging.info("Face encodings initialized successfully.")

# Start a separate thread to load encodings after Flask restarts
def start_load_encodings_thread():
    while True:
        # Ensure the encodings are loaded before requests come in
        time.sleep(1)
        if not encodings_initialized:
            load_encodings_on_restart()
            break
# Ensure that encodings are loaded after server restarts
if __name__ == '__main__':
    # Start thread to load encodings after server restart
    threading.Thread(target=start_load_encodings_thread, daemon=True).start()
    app.run(debug=True)
