from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import face_recognition
import cv2
import os
import numpy as np
from PIL import Image
import pyodbc
import logging
from sklearn.neighbors import NearestNeighbors

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configure paths
UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = r'C:\Users\dell\source\repos\PatientSystem\images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.patient_data = {}
        self.knn = None

    def connect_to_database(self):
        try:
            connection = pyodbc.connect(
                'DRIVER={SQL Server};'
                'SERVER=DESKTOP-CLQGA5Q\\SQLEXPRESS;'
                'DATABASE=PatientSystemDB;'
                'Trusted_Connection=yes;'
            )
            return connection
        except Exception as e:
            logging.error(f"Database connection error: {e}")
            return None

    def load_encoding_images(self, base_url):
        logging.info("Loading face encodings...")
        connection = self.connect_to_database()
        if connection is None:
            return

        try:
            cursor = connection.cursor()
            cursor.execute("SELECT Id, Name, Dob, Mobileno, Nationalno, FaceImg FROM dbo.Patients")
            patients = cursor.fetchall()

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

                # Skip unnecessary processing of image; just convert .png to .jpg directly
                if image_file.lower().endswith(".jpg"):
                    compressed_image_path = image_file.replace(".png", ".jpg")
                    with Image.open(image_file) as img:
                        img = img.convert("RGB")
                        img.save(compressed_image_path, "JPEG", quality=75)
                    image_file = compressed_image_path

                # Now load the image for encoding
                img = cv2.imread(image_file)
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

            if encodings:  # Ensure encodings are not empty
                self.knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
                self.knn.fit(encodings)

                self.known_face_encodings = encodings
                self.known_face_names = names
                self.patient_data = patient_metadata

                logging.info(f"Loaded {len(self.known_face_encodings)} face encodings.")
            else:
                logging.warning("No face encodings were loaded.")

        finally:
            connection.close()

    def compare_faces(self, unknown_image):
        unknown_encoding = face_recognition.face_encodings(unknown_image)
        if not unknown_encoding:
            logging.warning("No face found in the image.")
            return None

        unknown_encoding = unknown_encoding[0]
        distances, indices = self.knn.kneighbors([unknown_encoding])

        if distances[0][0] < 0.6:
            matched_name = self.known_face_names[indices[0][0]]
            return matched_name
        return None

@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

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

    # Skip unnecessary PNG to JPG compression
    if file_path.lower().endswith(".png"):
        compressed_file_path = file_path.replace(".png", ".jpg")
        with Image.open(file_path) as img:
            img = img.convert("RGB")
            img.save(compressed_file_path, "JPEG", quality=75)
        os.remove(file_path)
        file_path = compressed_file_path

    # Load image for face recognition
    unknown_image = face_recognition.load_image_file(file_path)

    # Load face encodings
    base_url = f"{request.scheme}://{request.host}"
    facerec = SimpleFacerec()
    facerec.load_encoding_images(base_url)

    # Ensure knn model is initialized and encodings are loaded
    if not facerec.knn:
        logging.error("KNN model is not initialized or no face encodings are loaded.")
        return jsonify({"error": "Face encodings are not available."}), 500

    matched_name = facerec.compare_faces(unknown_image)
    os.remove(file_path)

    if matched_name:
        patient_data = facerec.patient_data.get(matched_name)
        return jsonify({"isMatch": True, "patientName": matched_name, "patientData": patient_data})

    return jsonify({"isMatch": False, "patientName": "Unknown", "patientData": {}})

if __name__ == '__main__':
    app.run(debug=True)
