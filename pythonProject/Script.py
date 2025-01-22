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

# Configure the logging
logging.basicConfig(level=logging.INFO)

# Set up Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})

# Paths
IMAGE_FOLDER = r"C:\Users\dell\source\repos\PatientSystem\images"
ENCODING_FOLDER = r"C:\Users\dell\source\repos\PatientSystem\EncodingFile"
UPLOAD_FOLDER = r"C:\Users\dell\source\repos\PatientSystem\uploads"
base_url = "http://localhost:5000"

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
                'SERVER=DESKTOP-CLQGA5Q\\SQLEXPRESS;'
                'DATABASE=PatientSystemDB;'
                'Trusted_Connection=yes;'
            )
            return connection
        except Exception as e:
            logging.error(f"Database connection error: {e}")
            return None

    def load_encoding_images(self):
        logging.info("Loading face encodings...")
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
                encoding_file_path = os.path.join(ENCODING_FOLDER, f"{patient_id}_encoding.dat")

                if not os.path.exists(encoding_file_path):
                    logging.warning(f"Encoding file not found for patient {name}. Skipping.")
                    continue

                with open(encoding_file_path, 'rb') as encoding_file:
                    encoding = pickle.load(encoding_file)

                encodings.append(encoding)
                names.append(name)
                face_img_url = f"{base_url}/images/{os.path.basename(face_img)}"
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
        try:
            # Resize the image to reduce processing time
            small_image = cv2.resize(unknown_image, (320, 240))

            # Generate or load the encoding for the unknown image
            encodings = face_recognition.face_encodings(small_image)
            if len(encodings) == 0:
                logging.warning("No face found in the image.")
                return None  # No face found
            unknown_encoding = encodings[0]

            # Ensure the KNN model is initialized
            if not self.knn:
                logging.error("KNN model is not initialized.")
                return None

            # Find the closest match using KNN
            distances, indices = self.knn.kneighbors([unknown_encoding], n_neighbors=1)

            # Log the distance for debugging
            logging.info(f"Match distance: {distances[0][0]}")

            if distances[0][0] >= tolerance:
                logging.info("No match found within the tolerance threshold.")
                return None  # No match found

            # Return the matched name
            return self.known_face_names[indices[0][0]]
        except Exception as e:
            logging.error(f"Error comparing faces: {e}")
            return None

    def generate_encoding_file(self, patient_id, face_image_path):
        try:
            face_image = face_recognition.load_image_file(face_image_path)
            face_encoding = face_recognition.face_encodings(face_image)

            if face_encoding:
                encoding_file_path = os.path.join(ENCODING_FOLDER, f"{patient_id}_encoding.dat")
                with open(encoding_file_path, 'wb') as encoding_file:
                    pickle.dump(face_encoding[0], encoding_file)

                logging.info(f"Encoding file for patient {patient_id} saved at {encoding_file_path}")
                return encoding_file_path
            else:
                logging.error(f"No face found in the image for patient {patient_id}.")
                return None
        except Exception as e:
            logging.error(f"Error generating encoding file for patient {patient_id}: {e}")
            return None


# Flask routes
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

    unknown_image = face_recognition.load_image_file(file_path)

    try:
        matched_name = facerec.compare_faces(unknown_image, tolerance=0.6)
    finally:
        os.remove(file_path)

    if matched_name:
        patient_data = facerec.patient_data.get(matched_name)
        return jsonify({"isMatch": True, "patientName": matched_name, "patientData": patient_data})

    return jsonify({"isMatch": False, "patientName": "Unknown", "patientData": {}})


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


@app.route('/generate_encoding', methods=['POST'])
def generate_encoding():
    data = request.get_json()

    patient_id = data.get('patientId')
    face_image_path = data.get('faceImage')

    if not os.path.exists(face_image_path):
        return jsonify({"error": f"Face image not found at {face_image_path}"}), 400

    encoding_file_path = facerec.generate_encoding_file(patient_id, face_image_path)

    if encoding_file_path:
        return jsonify({"encodingFilePath": encoding_file_path}), 200
    else:
        return jsonify({"error": "Failed to generate encoding."}), 500


# Initialize the SimpleFacerec instance
facerec = SimpleFacerec()


# Load encodings on server restart
def load_encodings_on_restart():
    facerec.load_encoding_images()


if __name__ == '__main__':
    threading.Thread(target=load_encodings_on_restart, daemon=True).start()
    app.run(debug=True)
