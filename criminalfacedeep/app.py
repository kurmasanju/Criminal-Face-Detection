import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Configuration
DATASET_PATH = "dataset"
TRAINER_FILE = "trainer.yml"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if os.path.exists(TRAINER_FILE):
    recognizer.read(TRAINER_FILE)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_and_save_face(image_path, criminal_id, name, count):
    """Process image to extract face and save in LBPH format"""
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return False
    
    # Take only the first face found
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    # Resize to consistent dimensions
    face_roi = cv2.resize(face_roi, (200, 200))
    
    # Save the processed face
    filename = f"{DATASET_PATH}/{criminal_id}_{name}_{count}.jpg"
    cv2.imwrite(filename, face_roi)
    return True

def detect_faces():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("⚠️ Error: Camera not detected!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 60:
                text = f"ID: {face_id}"
                color = (0, 255, 0)  # Green for recognized faces
            else:
                text = "Unknown"
                color = (0, 0, 255)  # Red for unknown faces

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_criminal', methods=['GET', 'POST'])
def add_criminal():
    if request.method == 'POST':
        criminal_id = request.form['criminal_id']
        name = request.form['name']
        
        if not criminal_id or not name:
            flash("Criminal ID and Name are required!", "error")
            return redirect(url_for('add_criminal'))
        
        capture_images(criminal_id, name)
        flash(f"Criminal {name} added successfully!", "success")
        return redirect(url_for('index'))

    return render_template('add_criminal.html')

def capture_images(criminal_id, name):
    cap = cv2.VideoCapture(0)
    count = 0

    while count < 30:  # Capture 30 images
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            filename = f"{DATASET_PATH}/{criminal_id}_{name}_{count}.jpg"
            cv2.imwrite(filename, face_roi)
            count += 1

        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(100) == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    train_recognizer()

@app.route('/upload_images', methods=['GET', 'POST'])
def upload_images():
    if request.method == 'POST':
        criminal_id = request.form.get('criminal_id')
        name = request.form.get('name')
        files = request.files.getlist('images')

        if not criminal_id or not name:
            flash("Criminal ID and Name are required!", "error")
            return redirect(url_for('upload_images'))

        if len(files) == 0 or files[0].filename == '':
            flash("No selected files!", "error")
            return redirect(url_for('upload_images'))

        success_count = 0
        for i, file in enumerate(files):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(temp_path)
                
                if process_and_save_face(temp_path, criminal_id, name, i):
                    success_count += 1
                
                os.remove(temp_path)

        if success_count > 0:
            train_recognizer()
            flash(f"Successfully processed {success_count} images for {name}!", "success")
        else:
            flash("No valid faces found in the uploaded images!", "error")

        return redirect(url_for('index'))

    return render_template('upload_images.html')

def train_recognizer():
    faces = []
    labels = []
    
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(".jpg"):
            path = os.path.join(DATASET_PATH, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            label = int(filename.split("_")[0])
            faces.append(img)
            labels.append(label)

    if len(faces) == 0:
        print("⚠️ No images found in dataset!")
        return

    recognizer.train(faces, np.array(labels))
    recognizer.write(TRAINER_FILE)
    print("✅ Model trained successfully!")

if __name__ == '__main__':
    app.run(debug=True)