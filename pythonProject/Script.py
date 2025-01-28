import os
import logging
import threading
import pyodbc
import cv2
import face_recognition
import pickle
from sklearn.neighbors import NearestNeighbors
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})

# Configuration (can be moved to environment variables)
IMAGE_FOLDER = r"C:\Users\dell\source\repos\PatientSystem\images"
ENCODING_FOLDER = r"C:\Users\dell\source\repos\PatientSystem\EncodingFile"
UPLOAD_FOLDER = r"C:\Users\dell\source\repos\PatientSystem\uploads"
BASE_URL = "http://localhost:5000"
DATABASE_CONFIG = {
    'DRIVER': '{SQL Server}',
    'SERVER': 'DESKTOP-CLQGA5Q\\SQLEXPRESS',
    'DATABASE': 'PatientSystemDB',
    'Trusted_Connection': 'yes'
}

# Ensure directories exist
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(ENCODING_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the SimpleFacerec class
class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.patient_data = {}
        self.knn = None
        self.lock = threading.Lock()

    def connect_to_database(self):
        """Establish a connection to the database."""
        try:
            connection = pyodbc.connect(**DATABASE_CONFIG)
            return connection
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return None

    def load_encoding_images(self):
        """Load face encodings from the database and encoding files."""
        logger.info("Loading face encodings...")
        connection = self.connect_to_database()
        if connection is None:
            return

        try:
            cursor = connection.cursor()
            cursor.execute("SELECT Id, Name, Dob, Mobileno, Nationalno, FaceImg FROM dbo.Patients")
            patients = cursor.fetchall()

            if not patients:
                logger.warning("No patient data found in the database.")
                return

            encodings = []
            names = []
            patient_metadata = {}

            for patient in patients:
                patient_id, name, dob, mobile_no, national_no, face_img = patient
                encoding_file_path = os.path.join(ENCODING_FOLDER, f"{patient_id}_encoding.dat")

                if not os.path.exists(encoding_file_path):
                    logger.error(f"Encoding file not found for patient {name} (ID: {patient_id}). Skipping.")
                    continue

                with open(encoding_file_path, 'rb') as encoding_file:
                    encoding = pickle.load(encoding_file)

                encodings.append(encoding)
                names.append(name)
                face_img_url = f"{BASE_URL}/images/{os.path.basename(face_img)}"
                patient_metadata[name] = {
                    "mobileno": mobile_no,
                    "nationalno": national_no,
                    "id": patient_id,
                    "dob": dob,
                    "faceImg": face_img_url
                }

            if encodings:
                with self.lock:
                    self.knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
                    self.knn.fit(encodings)
                    self.known_face_encodings = encodings
                    self.known_face_names = names
                    self.patient_data = patient_metadata
                    logger.info(f"Loaded {len(self.known_face_encodings)} face encodings.")
            else:
                logger.warning("No face encodings were loaded.")

        except Exception as e:
            logger.error(f"Error loading encodings: {e}")
        finally:
            connection.close()

    def compare_faces(self, unknown_image, tolerance=0.5):
        """Compare an unknown face image with known encodings."""
        try:
            small_image = cv2.resize(unknown_image, (320, 240))
            encodings = face_recognition.face_encodings(small_image)

            if not encodings:
                logger.warning("No face found in the image.")
                return None

            unknown_encoding = encodings[0]

            with self.lock:
                if not self.knn:
                    logger.error("KNN model is not initialized.")
                    return None

                distances, indices = self.knn.kneighbors([unknown_encoding], n_neighbors=1)
                logger.info(f"Match distance: {distances[0][0]}")

                if distances[0][0] >= tolerance:
                    logger.info("No match found within the tolerance threshold.")
                    return None

                return self.known_face_names[indices[0][0]]

        except Exception as e:
            logger.error(f"Error comparing faces: {e}")
            return None

    def generate_encoding_file(self, patient_id, face_image_path):
        """Generate and save an encoding file for a given face image."""
        try:
            face_image = face_recognition.load_image_file(face_image_path)
            face_encoding = face_recognition.face_encodings(face_image)

            if not face_encoding:
                logger.error(f"No face found in the image for patient {patient_id}.")
                return None

            encoding_file_path = os.path.join(ENCODING_FOLDER, f"{patient_id}_encoding.dat")
            with open(encoding_file_path, 'wb') as encoding_file:
                pickle.dump(face_encoding[0], encoding_file)

            with self.lock:
                self.known_face_encodings.append(face_encoding[0])
                self.known_face_names.append(patient_id)
                if self.knn:
                    self.knn.fit(self.known_face_encodings)

            logger.info(f"Encoding file for patient {patient_id} saved at {encoding_file_path}")
            return encoding_file_path

        except Exception as e:
            logger.error(f"Error generating encoding file for patient {patient_id}: {e}")
            return None


# Flask routes
@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory(IMAGE_FOLDER, filename)


@app.route('/detectAndFind', methods=['POST'])
def detect_and_find():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        unknown_image = face_recognition.load_image_file(file_path)
        matched_name = facerec.compare_faces(unknown_image, tolerance=0.5)

        if matched_name:
            patient_data = facerec.patient_data.get(matched_name)
            return jsonify({"status": "success", "isMatch": True, "patientName": matched_name, "patientData": patient_data})
        else:
            return jsonify({"status": "success", "isMatch": False, "patientName": "Unknown", "patientData": {}})

    except Exception as e:
        logger.error(f"Error in /detectAndFind endpoint: {e}")
        return jsonify({"status": "error", "message": "An error occurred.", "details": str(e)}), 500
    finally:
        os.remove(file_path)


@app.route('/add_encoding_to_database', methods=['POST'])
def add_encoding_to_database():
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")

        patient_id = data.get('patientId')
        face_image_path = data.get('faceImage')

        if not patient_id or not face_image_path:
            return jsonify({"status": "error", "message": "Missing patientId or faceImage path."}), 400

        if not os.path.exists(face_image_path):
            return jsonify({"status": "error", "message": f"Face image not found at {face_image_path}"}), 400

        encoding_file_path = facerec.generate_encoding_file(patient_id, face_image_path)
        if not encoding_file_path:
            return jsonify({"status": "error", "message": "Failed to generate encoding file."}), 500

        connection = facerec.connect_to_database()
        if not connection:
            return jsonify({"status": "error", "message": "Database connection failed."}), 500

        try:
            cursor = connection.cursor()
            cursor.execute("UPDATE dbo.Patients SET EncodingFile = ? WHERE Id = ?", encoding_file_path, patient_id)
            connection.commit()
            logger.info(f"Database updated for patientId {patient_id} with encoding file path: {encoding_file_path}")
            return jsonify({"status": "success", "message": "Encoding file generated and database updated successfully.", "encodingFilePath": encoding_file_path}), 200
        except Exception as e:
            logger.error(f"Error updating database: {e}")
            return jsonify({"status": "error", "message": "Failed to update database.", "details": str(e)}), 500
        finally:
            connection.close()

    except Exception as e:
        logger.error(f"Error in /add_encoding_to_database endpoint: {e}")
        return jsonify({"status": "error", "message": "An error occurred.", "details": str(e)}), 500


@app.route('/reload_encodings', methods=['GET'])
def reload_encodings():
    try:
        facerec.load_encoding_images()
        return jsonify({"status": "success", "message": "Encodings reloaded successfully."}), 200
    except Exception as e:
        logger.error(f"Error during reload: {e}")
        return jsonify({"status": "error", "message": "Failed to reload encodings.", "details": str(e)}), 500


@app.route('/generate_encoding', methods=['POST'])
def generate_encoding():
    data = request.get_json()
    patient_id = data.get('patientId')
    face_image_path = data.get('faceImage')

    if not os.path.exists(face_image_path):
        return jsonify({"status": "error", "message": f"Face image not found at {face_image_path}"}), 400

    encoding_file_path = facerec.generate_encoding_file(patient_id, face_image_path)
    if encoding_file_path:
        return jsonify({"status": "success", "encodingFilePath": encoding_file_path}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to generate encoding."}), 500


# Initialize the SimpleFacerec instance
facerec = SimpleFacerec()

# Load encodings on server restart
def load_encodings_on_restart():
    facerec.load_encoding_images()


if __name__ == '__main__':
    threading.Thread(target=load_encodings_on_restart, daemon=True).start()
    app.run(debug=True)