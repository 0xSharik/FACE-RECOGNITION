import cv2
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import base64
import uuid
import shutil
from functools import wraps
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_cors import CORS
import logging
from datetime import datetime
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# Use environment variable for storage path (Railway Volume)
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', '/data/known_faces')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ThingSpeak API key (Optional: Keep if user wants it)
WRITE_API_KEY = "96A81RBRK83RRP5I"
THINGSPEAK_URL = "https://api.thingspeak.com/update"

# Model Configuration
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface" # More accurate
RECOGNITION_THRESHOLD = 0.68 

# Global Service State
is_service_active = True

# Cache
known_encodings = []
known_names = []
last_cache_update = 0

def load_known_faces(force=False):
    """
    Uses a simple memory cache to avoid reloading on every request.
    """
    global known_encodings, known_names, last_cache_update

    # Check if folder was modified
    try:
        current_mtime = os.path.getmtime(app.config['UPLOAD_FOLDER'])
    except FileNotFoundError:
        current_mtime = 0

    if not force and current_mtime <= last_cache_update and known_encodings:
        return # Cache is valid

    logger.info("Reloading face encodings...")
    temp_encodings = []
    temp_names = []
    
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # 1. Process Folders (New Structure)
    for entry in os.listdir(app.config['UPLOAD_FOLDER']):
        entry_path = os.path.join(app.config['UPLOAD_FOLDER'], entry)
        
        if os.path.isdir(entry_path):
            person_name = entry
            # Go through images in this person's folder
            image_files = [f for f in os.listdir(entry_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            for img_file in image_files:
                img_path = os.path.join(entry_path, img_file)
                try:
                    embeddings = DeepFace.represent(
                        img_path=img_path,
                        model_name=MODEL_NAME,
                        enforce_detection=False,
                        detector_backend=DETECTOR_BACKEND
                    )
                    if embeddings:
                        temp_encodings.append(embeddings[0]["embedding"])
                        temp_names.append(person_name)
                except Exception as e:
                    logger.error(f"Error loading {img_file} for {person_name}: {e}")

        # 2. Process Flat Files (Legacy Support)
        elif os.path.isfile(entry_path) and entry.lower().endswith(('.jpg', '.png', '.jpeg')):
            filename = entry
            person_name = os.path.splitext(filename)[0]
            try:
                embeddings = DeepFace.represent(
                    img_path=entry_path,
                    model_name=MODEL_NAME,
                    enforce_detection=False,
                    detector_backend=DETECTOR_BACKEND
                )
                if embeddings:
                    temp_encodings.append(embeddings[0]["embedding"])
                    temp_names.append(person_name)
            except Exception as e:
                logger.error(f"Error loading legacy file {filename}: {e}")

    known_encodings = temp_encodings
    known_names = temp_names
    last_cache_update = time.time()
    logger.info(f"Loaded {len(known_names)} face vectors for {len(set(known_names))} unique people.")

def send_to_thingspeak(name):
    # (Simplified for cloud - usually we want this async to not block the request)
    # For now, suppressing to ensure fast response, or enable if critical.
    pass 

# --- Security Decorator ---
def require_admin(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        admin_token = os.environ.get('ADMIN_TOKEN')
        if admin_token:
            token = request.headers.get('X-ADMIN-TOKEN')
            if not token or token != admin_token:
                return jsonify({'error': 'Unauthorized', 'message': 'Invalid or missing Admin Token'}), 401
        return f(*args, **kwargs)
    return decorated_function

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    # Reload faces to ensure list is up to date
    load_known_faces() 
    return render_template('admin.html', faces=known_names)

# --- ATTRIBUTES ---
sessions = {} # {sid: {last_seen, name, status, warnings...}}
session_frames = {} # {sid: small_jpeg_bytes}

def cleanup_sessions():
    now = time.time()
    to_remove = []
    for sid, data in sessions.items():
        # datetime string to timestamp
        last_seen = datetime.fromisoformat(data['last_seen']).timestamp()
        if now - last_seen > 30:
            to_remove.append(sid)
    
    for sid in to_remove:
        sessions.pop(sid, None)
        session_frames.pop(sid, None)

@app.route('/api/recognize', methods=['POST'])
def recognize():
    global is_service_active
    if not is_service_active:
         return jsonify({'status': 'stopped', 'message': 'Service is paused'}), 503

    data = request.get_json()
    if not data or 'image' not in data:
         return jsonify({'status': 'error', 'message': 'No image data'}), 400
    
    # Get Session ID (Essential for Admin)
    sid = request.args.get('session_id') or data.get('session_id')
    if not sid:
         # Fallback for legacy calls without ID (create one or ignore?)
         # For now, generate temporary one or fail? 
         # Let's fail or the admin panel won't work.
         # But better to handle gracefully.
         sid = "legacy_session"

    start_time = time.time()
    
    # Periodic Cleanup
    if int(start_time) % 10 == 0:
        cleanup_sessions()

    try:
        # Decode image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
             return jsonify({'status': 'error', 'message': 'Bad frame'}), 400

        # FIX 1: Decouple Admin Frame (Downscale & Compress)
        # Resize to 320x240 for Admin UI
        try:
            admin_small = cv2.resize(frame, (320, 240))
            _, admin_jpeg = cv2.imencode('.jpg', admin_small, [cv2.IMWRITE_JPEG_QUALITY, 40])
            session_frames[sid] = admin_jpeg.tobytes()
        except Exception:
            pass # Don't detect if resize fails

        # FIX: DeepFace Logic (Only run if service active)
        load_known_faces()
        
        try:
            target_embeddings = DeepFace.represent(
                img_path=frame,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True
            )
        except ValueError:
             # Update Session (No Face)
             update_session(sid, "no_face", "Unknown", 0)
             return jsonify({'status': 'no_face', 'message': 'No face detected'}), 200
        except Exception as e:
             return jsonify({'status': 'error', 'message': str(e)}), 500

        if len(target_embeddings) > 1:
             update_session(sid, "multiple_faces", "Unknown", 0)
             return jsonify({'status': 'multiple_faces', 'message': 'Multiple faces'}), 200

        match_name = "Unknown"
        match_dist = 1.0
        
        if target_embeddings and known_encodings:
            target_vector = target_embeddings[0]["embedding"]
            target_np = np.array(target_vector).reshape(1, -1)
            known_np = np.array(known_encodings)
            
            similarities = cosine_similarity(target_np, known_np)[0]
            distances = 1 - similarities
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            
            if min_dist < RECOGNITION_THRESHOLD:
                match_name = known_names[min_idx]
                match_dist = min_dist

        # Update Session with Result
        status = "verified" if match_name != "Unknown" else "unstable_identity"
        update_session(sid, status, match_name, (1-match_dist)*100)

        return jsonify({
            'status': 'success',
            'name': match_name,
            'distance': float(match_dist),
            'process_time': time.time() - start_time
        })

    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def update_session(sid, status, name, conf):
    now_str = datetime.now().isoformat()
    if sid not in sessions:
        sessions[sid] = {
            'start_time': now_str,
            'total_warnings': 0,
            'warnings': []
        }
    
    sessions[sid]['last_seen'] = now_str
    sessions[sid]['status'] = status
    sessions[sid]['current_name'] = name
    sessions[sid]['confidence'] = round(conf, 1)

    # Simple logic to increment warnings for this demo
    if status in ['no_face', 'multiple_faces']:
        sessions[sid]['total_warnings'] += 1

@app.route('/api/admin/stats', methods=['GET'])
@require_admin
def admin_stats():
    cleanup_sessions() # Ensure clean before showing
    return jsonify({
        'active_count': len(sessions),
        'sessions': sessions
    })

@app.route('/api/admin/frame/<sid>', methods=['GET'])
@require_admin
def admin_frame(sid):
    # FIX 3: Best Effort - No Blocking, No Processing
    blob = session_frames.get(sid)
    if not blob:
        return "", 204 # No Content
    return (blob, 200, {'Content-Type': 'image/jpeg'})



# --- Service Control API ---

@app.route('/api/service_status', methods=['GET'])
def service_status():
    return jsonify({'active': is_service_active})

@app.route('/api/toggle_service', methods=['POST'])
@require_admin
def toggle_service():
    global is_service_active
    data = request.get_json()
    action = data.get('action') # 'start' or 'stop'
    
    if action == 'start':
        is_service_active = True
    elif action == 'stop':
        is_service_active = False
        
    logger.info(f"Service state changed to: {is_service_active}")
    return jsonify({'success': True, 'active': is_service_active})

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'faces_loaded': len(set(known_names)),
        'vectors_loaded': len(known_encodings),
        'service_active': is_service_active,
        'backend': DETECTOR_BACKEND
    })

# --- Admin API ---

@app.route('/api/upload_face', methods=['POST'])
@require_admin
def upload_face():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    name = request.form.get('name', '').strip()
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and name:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png']:
             return jsonify({'error': 'Invalid file type'}), 400
             
        safe_name = secure_filename(name)
        
        # Create folder for person if not exists
        person_dir = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Save file with unique ID
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(person_dir, unique_filename)
        
        file.save(save_path)
        
        # Invalidate cache
        global last_cache_update
        last_cache_update = 0
        
        return jsonify({'success': True, 'message': f'Added image for {name}'})
    
    return jsonify({'error': 'Missing name or file'}), 400

@app.route('/api/delete_face', methods=['POST'])
@require_admin
def delete_face():
    data = request.get_json()
    name = data.get('name')
    
    if not name:
        return jsonify({'error': 'No name provided'}), 400
        
    safe_name = secure_filename(name)
    target_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    deleted = False

    # Try deleting folder
    if os.path.isdir(target_path):
        try:
            shutil.rmtree(target_path)
            deleted = True
        except Exception as e:
            logger.error(f"Error deleting folder {target_path}: {e}")
            return jsonify({'error': f"Failed to delete {name}"}), 500
            
    # Fallback: Try deleting legacy file
    if not deleted:
        # Check for any file starting with name
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            if os.path.splitext(f)[0] == safe_name and os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f)):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
                deleted = True
                break

    if deleted:
        global last_cache_update
        last_cache_update = 0
        return jsonify({'success': True, 'message': f'Deleted {name}'})
    else:
        return jsonify({'error': 'Face not found'}), 404

if __name__ == "__main__":
    # Warm up model to prevent timeout on first request
    print("Scalping Neural Engine (Warming up DeepFace)...")
    try:
        DeepFace.build_model(MODEL_NAME)
        print("Neural Engine Online.")
    except Exception as e:
        print(f"Model load warning: {e}")

    # Load faces initially
    load_known_faces(force=True)
    
    # Get PORT from env (Railway) or default to 5000
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Face Recognition Server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
