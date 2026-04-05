"""
========================================
NUMBER PLATE DETECTION SYSTEM - PHASE 3
Flask Web App + SQLite + Blacklist Alert
========================================
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from datetime import datetime
import base64
import database as db

# Try to import easyocr, fallback to mock if unavailable
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    OCR_AVAILABLE = True
    print("[INFO] EasyOCR loaded successfully!")
except Exception as e:
    print(f"[WARNING] EasyOCR not available: {e}")
    OCR_AVAILABLE = False
    reader = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(filtered, 30, 200)
    return gray, filtered, edges

def find_plate_contour(edges):
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        if len(approx) == 4:
            return approx
    return None

def extract_plate(image, contour):
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(contour)
    plate_crop = masked[y:y+h, x:x+w]
    return plate_crop, (x, y, w, h)

def read_plate_text(plate_image):
    if not OCR_AVAILABLE or reader is None:
        return "DEMO-PLATE-XY01", 92.5
    try:
        rgb_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb_plate)
        plate_text = ""
        confidence = 0.0
        for (bbox, text, prob) in results:
            if prob > 0.2:
                plate_text += text + " "
                confidence = max(confidence, prob)
        return plate_text.strip().upper(), round(confidence * 100, 2)
    except Exception as e:
        print(f"[OCR Error] {e}")
        return "", 0.0

def draw_results(image, contour, plate_text, bbox, confidence):
    output = image.copy()
    cv2.drawContours(output, [contour], -1, (0, 255, 0), 3)
    x, y, w, h = bbox
    cv2.rectangle(output, (x, max(0, y - 45)), (x + w, y), (0, 200, 0), -1)
    cv2.putText(output, plate_text, (x + 4, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
    cv2.putText(output, f"Conf: {confidence}%", (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(output, ts, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    return output

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def process_image_file(filepath):
    """Full detection pipeline — returns dict with all results."""
    image = cv2.imread(filepath)
    if image is None:
        return {"success": False, "error": "Could not load image"}

    image = cv2.resize(image, (800, 600))
    gray, filtered, edges = preprocess_image(image)
    contour = find_plate_contour(edges)

    if contour is None:
        return {
            "success": False,
            "error": "No number plate detected in image",
            "original_img": image_to_base64(image)
        }

    plate_crop, bbox = extract_plate(image, contour)

    if plate_crop.size == 0:
        return {"success": False, "error": "Plate extraction failed"}

    plate_text, confidence = read_plate_text(plate_crop)

    if not plate_text:
        plate_text = "UNREADABLE"
        confidence = 0.0

    result_image = draw_results(image, contour, plate_text, bbox, confidence)

    # Check blacklist
    is_blacklisted = db.is_blacklisted(plate_text)

    # Save to database
    db.save_detection(
        plate_text=plate_text,
        confidence=confidence,
        image_path=os.path.basename(filepath),
        is_blacklisted=is_blacklisted
    )

    return {
        "success": True,
        "plate_text": plate_text,
        "confidence": confidence,
        "is_blacklisted": is_blacklisted,
        "original_img": image_to_base64(image),
        "result_img": image_to_base64(result_image),
        "plate_img": image_to_base64(plate_crop),
        "bbox": list(bbox)
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    stats = db.get_stats()
    recent = db.get_recent_detections(5)
    return render_template('index.html', stats=stats, recent=recent)

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"})
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_")
            filename = ts + filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = process_image_file(filepath)
            return jsonify(result)
        return jsonify({"success": False, "error": "Invalid file type. Use JPG/PNG."})
    return render_template('detect.html')

@app.route('/history')
def history():
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '')
    detections = db.get_all_detections(page=page, search=search)
    total = db.get_total_count(search=search)
    total_pages = (total + 19) // 20
    return render_template('history.html',
                           detections=detections,
                           page=page,
                           total_pages=total_pages,
                           search=search,
                           total=total)

@app.route('/blacklist', methods=['GET'])
def blacklist():
    plates = db.get_blacklist()
    return render_template('blacklist.html', plates=plates)

@app.route('/blacklist/add', methods=['POST'])
def add_blacklist():
    plate = request.form.get('plate', '').upper().strip()
    reason = request.form.get('reason', '').strip()
    if plate:
        db.add_to_blacklist(plate, reason)
    return redirect(url_for('blacklist'))

@app.route('/blacklist/remove/<int:plate_id>', methods=['POST'])
def remove_blacklist(plate_id):
    db.remove_from_blacklist(plate_id)
    return redirect(url_for('blacklist'))

@app.route('/stats')
def stats():
    data = db.get_full_stats()
    return render_template('stats.html', data=data)

@app.route('/api/stats')
def api_stats():
    return jsonify(db.get_full_stats())

@app.route('/api/recent')
def api_recent():
    detections = db.get_recent_detections(10)
    return jsonify([dict(d) for d in detections])

@app.route('/delete/<int:det_id>', methods=['POST'])
def delete_detection(det_id):
    db.delete_detection(det_id)
    return redirect(url_for('history'))

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    db.init_db()
    db.seed_demo_data()
    print("\n" + "="*50)
    print("  NUMBER PLATE DETECTION - PHASE 3")
    print("  Open browser: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
