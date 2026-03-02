from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_cors import CORS  
from werkzeug.utils import secure_filename
import cv2, pickle, numpy as np, tensorflow as tf
from utils_preproc_two import pack_two_hands, smooth_labels
import mediapipe as mp
from collections import deque
import os
import json

# === Load model & encoder ===
MODEL_PATH = "model/isl_model_two.keras"
ENCODER_PATH = "model/label_encoder_two.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Admin password (change this to your desired password)
ADMIN_PASSWORD = "admin123"

# Create directories for help images
HELP_IMAGES_DIR = "static/help_images"
for category in ["numbers", "alphabets", "words"]:
    os.makedirs(os.path.join(HELP_IMAGES_DIR, category), exist_ok=True)

# === Mediapipe hands ===
mp_hands = mp.solutions.hands
# Optimize MediaPipe for faster processing
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5,
                       model_complexity=0)  # Use simpler model for speed (0=fastest, 2=most accurate)
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
# Optimize video capture settings for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
# Reduce buffer size for faster response
buffer = deque(maxlen=5)
latest_label = None

# Control flags
predicting = False

def gen_frames():
    global predicting
    global latest_label
    while True:
        if not predicting:
            frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Prediction Stopped", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            # Optimize JPEG encoding for faster streaming
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            ret, buffer_ = cv2.imencode('.jpg', frame, encode_params)
            frame_bytes = buffer_.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue

        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        h, w = frame.shape[:2]
        hands_dict = {'Left': None, 'Right': None}
        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                lr = handed.classification[0].label
                pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
                hands_dict[lr] = pts
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                                       mp_style.get_default_hand_landmarks_style(),
                                       mp_style.get_default_hand_connections_style())

        # === Only predict if at least one hand is detected ===
        if hands_dict['Left'] is not None or hands_dict['Right'] is not None:
            feats, mask = pack_two_hands(hands_dict)
            inp = np.concatenate([feats, mask], axis=0).reshape(1, -1)

            # Optimize prediction - use smaller batch and faster inference
            pred = model.predict(inp, verbose=0, batch_size=1)
            confidence = float(np.max(pred))
            cls = encoder.inverse_transform([np.argmax(pred)])[0]
            
            # Check confidence FIRST - show Unknown immediately if low confidence
            bar_height = 100
            cv2.rectangle(frame, (0, h-bar_height), (w, h), (0, 0, 0), -1)
            
            if confidence < 0.80:
                # Show Unknown Gesture immediately without waiting for buffer
                text = f"Unknown Gesture"
                cv2.putText(frame, text, (20, h-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
                latest_label = None
                buffer.clear()  # Clear buffer for unknown gestures
            else:
                # Only add to buffer if confidence is high
                buffer.append(cls)
                # Use smoothing only for high-confidence predictions
                label = smooth_labels(list(buffer), min_count=3)  # Reduced min_count for faster response
                if label:
                    text = f"Prediction: {label.upper()} ({confidence*100:.1f}%)"
                    cv2.putText(frame, text, (20, h-40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)
                    latest_label = label
                else:
                    # Buffer not ready yet, show current prediction with confidence
                    text = f"Prediction: {cls.upper()} ({confidence*100:.1f}%)"
                    cv2.putText(frame, text, (20, h-40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4, cv2.LINE_AA)
        else:
            # No hand detected
            buffer.clear()  # Clear buffer when no hands detected

        # Optimize JPEG encoding for faster streaming (slightly lower quality for speed)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]  # Reduce from default 95 to 85 for speed
        ret, buffer_ = cv2.imencode('.jpg', frame, encode_params)
        frame_bytes = buffer_.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    with open('simple_frontend.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/old')
def old():
    return render_template('index.html')

@app.route('/modern')
def modern():
    with open('simple_frontend.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start_prediction():
    global predicting
    predicting = True
    return jsonify({"status": "started"})

@app.route('/stop')
def stop_prediction():
    global predicting
    global latest_label
    predicting = False
    latest_label = None
    return jsonify({"status": "stopped"})

@app.route('/latest_label')
def get_latest_label():
    return jsonify({"label": latest_label})

@app.route('/exit')
def exit_app():
    cap.release()
    cv2.destroyAllWindows()
    os._exit(0)
    return "Exited"

# Help Images API Routes
@app.route('/api/images/<category>')
def get_images(category):
    """Get list of images for a category"""
    if category not in ["numbers", "alphabets", "words"]:
        return jsonify({"error": "Invalid category"}), 400
    
    category_dir = os.path.join(HELP_IMAGES_DIR, category)
    images = []
    
    if os.path.exists(category_dir):
        for filename in os.listdir(category_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                # Extract label from filename (format: label_filename.ext)
                label = filename.rsplit('_', 1)[0] if '_' in filename else filename.rsplit('.', 1)[0]
                images.append({
                    "filename": filename,
                    "label": label
                })
    
    # Sort by label
    images.sort(key=lambda x: x["label"])
    return jsonify({"images": images})

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Upload an image (admin only)"""
    password = request.form.get('password')
    if password != ADMIN_PASSWORD:
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    category = request.form.get('category')
    label = request.form.get('label')
    
    if category not in ["numbers", "alphabets", "words"]:
        return jsonify({"success": False, "message": "Invalid category"}), 400
    
    if not label:
        return jsonify({"success": False, "message": "Label is required"}), 400
    
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "message": "No file selected"}), 400
    
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
        # Create secure filename: label_originalname.ext
        original_filename = secure_filename(file.filename)
        label_clean = secure_filename(label)
        filename = f"{label_clean}_{original_filename}"
        
        category_dir = os.path.join(HELP_IMAGES_DIR, category)
        filepath = os.path.join(category_dir, filename)
        
        # If file exists, remove old one
        if os.path.exists(filepath):
            os.remove(filepath)
        
        file.save(filepath)
        return jsonify({"success": True, "message": "Image uploaded successfully", "filename": filename})
    
    return jsonify({"success": False, "message": "Invalid file type"}), 400

@app.route('/api/delete', methods=['POST'])
def delete_image():
    """Delete an image (admin only)"""
    data = request.get_json()
    password = data.get('password')
    
    if password != ADMIN_PASSWORD:
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    category = data.get('category')
    filename = data.get('filename')
    
    if category not in ["numbers", "alphabets", "words"]:
        return jsonify({"success": False, "message": "Invalid category"}), 400
    
    if not filename:
        return jsonify({"success": False, "message": "Filename is required"}), 400
    
    filepath = os.path.join(HELP_IMAGES_DIR, category, secure_filename(filename))
    
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({"success": True, "message": "Image deleted successfully"})
    
    return jsonify({"success": False, "message": "File not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
