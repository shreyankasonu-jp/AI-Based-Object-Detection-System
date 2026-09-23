from flask import Flask, render_template, request, session
from ultralytics import YOLO
import cv2
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Secret key for session
app.secret_key = "object_detection_secret"

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder automatically
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")


# ---------------- DATABASE ----------------

def init_db():

    conn = sqlite3.connect(
        'detections.db'
    )

    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS
    detection_history (

        id INTEGER
        PRIMARY KEY AUTOINCREMENT,

        image_name TEXT,

        objects_detected TEXT,

        confidence REAL,

        detection_time TEXT
    )
    ''')

    conn.commit()
    conn.close()


init_db()


# ---------------- LOGIN PAGE ----------------

@app.route('/')
def login():

    return render_template(
        'login.html'
    )


# ---------------- ABOUT PAGE ----------------

@app.route('/about')
def about():

    return render_template(
        'about.html'
    )


# ---------------- SIGNUP PAGE ----------------

@app.route('/signup')
def signup():

    return render_template(
        'signup.html'
    )


# ---------------- UPLOAD PAGE ----------------

@app.route('/upload')
def upload():

    upload_folder = app.config[
        'UPLOAD_FOLDER'
    ]

    image_count = len([

        img for img in
        os.listdir(upload_folder)

        if img.lower().endswith(
            ('.png', '.jpg', '.jpeg')
        )

        and img != 'result.jpg'
    ])

    return render_template(

        'upload.html',

        image=None,

        image_count=image_count
    )


# ---------------- DETECT OBJECTS ----------------

@app.route('/detect', methods=['POST'])
def detect():

    file = request.files['image']

    if file:

        filepath = os.path.join(

            app.config[
                'UPLOAD_FOLDER'
            ],

            file.filename
        )

        file.save(filepath)

        # Read image
        image = cv2.imread(
            filepath
        )

        # YOLO detection
        results = model(image)

        # Draw detection boxes
        output = results[0].plot()

        # Save result image
        result_path = os.path.join(

            app.config[
                'UPLOAD_FOLDER'
            ],

            'result.jpg'
        )

        cv2.imwrite(
            result_path,
            output
        )

        # ---------- STATISTICS ----------

        detected_objects = {}

        confidence_scores = []

        for box in results[0].boxes:

            class_id = int(
                box.cls[0]
            )

            object_name = model.names[
                class_id
            ]

            confidence = float(
                box.conf[0]
            ) * 100

            confidence_scores.append(
                round(
                    confidence,
                    2
                )
            )

            if object_name in detected_objects:

                detected_objects[
                    object_name
                ] += 1

            else:

                detected_objects[
                    object_name
                ] = 1

        total_objects = len(
            results[0].boxes
        )

        highest_confidence = (

            max(
                confidence_scores
            )

            if confidence_scores

            else 0
        )

        # ---------- TOTAL IMAGE COUNT ----------

        upload_folder = app.config[
            'UPLOAD_FOLDER'
        ]

        image_count = len([

            img for img in
            os.listdir(
                upload_folder
            )

            if img.lower().endswith(
                (
                    '.png',
                    '.jpg',
                    '.jpeg'
                )
            )

            and img != 'result.jpg'
        ])

        # ---------- SAVE TO DATABASE ----------

        objects_text = ", ".join([

            f"{obj}:{count}"

            for obj, count in
            detected_objects.items()

        ])

        conn = sqlite3.connect(
            'detections.db'
        )

        cursor = conn.cursor()

        cursor.execute('''
        INSERT INTO
        detection_history (

            image_name,

            objects_detected,

            confidence,

            detection_time
        )

        VALUES (?, ?, ?, ?)
        ''', (

            file.filename,

            objects_text,

            round(
                highest_confidence,
                2
            ),

            datetime.now().strftime(
                "%d-%m-%Y %H:%M"
            )
        ))

        conn.commit()
        conn.close()

        # ---------- SAVE DATA FOR DASHBOARD ----------

        session['image'] = (
            'uploads/result.jpg'
        )

        session['image_count'] = (
            image_count
        )

        session['object_count'] = (
            total_objects
        )

        session['confidence'] = round(
            highest_confidence,
            2
        )

        session[
            'detected_objects'
        ] = detected_objects

        # NEW: Detection Time
        session[
            'detection_time'
        ] = datetime.now().strftime(
            "%d-%m-%Y | %I:%M %p"
        )

        # ---------- RESULT PAGE ----------

        return render_template(

            'upload.html',

            image=
            'uploads/result.jpg',

            image_count=
            image_count
        )


# ---------------- DASHBOARD PAGE ----------------

@app.route('/dashboard')
def dashboard():

    conn = sqlite3.connect(
        'detections.db'
    )

    cursor = conn.cursor()

    cursor.execute('''
    SELECT *
    FROM detection_history
    ORDER BY id DESC
    LIMIT 5
    ''')

    history = cursor.fetchall()

    conn.close()

    return render_template(

        'dashboard.html',

        image=session.get(
            'image'
        ),

        image_count=session.get(
            'image_count',
            0
        ),

        object_count=session.get(
            'object_count',
            0
        ),

        confidence=session.get(
            'confidence',
            0
        ),

        detected_objects=session.get(
            'detected_objects',
            {}
        ),

        history=history,

        detection_time=session.get(
            'detection_time',
            'No Detection'
        )
    )


# ---------------- RUN APP ----------------

if __name__ == '__main__':
    app.run(debug=True)