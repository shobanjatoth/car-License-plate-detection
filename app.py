import os
import cv2
import torch
import uuid
import subprocess
import shutil
from flask import Flask, render_template, request, redirect
from paddleocr import PaddleOCR
from ultralytics import YOLO
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Configure upload folders
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# Path to FFmpeg (search system-wide)
FFMPEG_PATH = shutil.which("ffmpeg")
if FFMPEG_PATH is None:
    raise FileNotFoundError("FFmpeg is not installed or not found in system PATH")

# Load YOLOv8 model
model = YOLO("best.pt")  # Your trained YOLO model
ocr = PaddleOCR(use_angle_cls=True, lang="en")  # Initialize PaddleOCR

# Function to convert video for web playback
def convert_video_for_web(input_path, output_path):
    ffmpeg_cmd = [
        FFMPEG_PATH, "-i", input_path,
        "-vcodec", "libx264", "-acodec", "aac", "-strict", "experimental",
        "-movflags", "+faststart", "-preset", "slow", "-crf", "23", output_path
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Function to process images
def process_image(image_path):
    img = cv2.imread(image_path)
    results = model(image_path)
    detected_texts = set()

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        plate_img = img[y1:y2, x1:x2]
        if plate_img.size == 0:
            continue
        gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        ocr_result = ocr.ocr(gray_plate, cls=True)
        if ocr_result:
            for res in ocr_result:
                if res:
                    for line in res:
                        detected_texts.add(line[1][0])

    processed_image_path = os.path.join(PROCESSED_FOLDER, f"processed_{uuid.uuid4().hex}.jpg")
    cv2.imwrite(processed_image_path, img)
    return processed_image_path, list(detected_texts)

# Function to process videos
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detected_texts = set()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_path = os.path.join(PROCESSED_FOLDER, f"temp_{uuid.uuid4().hex}.mp4")
    out = cv2.VideoWriter(temp_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:
            results = model(frame)
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.tolist())
                plate_img = frame[y1:y2, x1:x2]
                if plate_img.size == 0:
                    continue
                gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                ocr_result = ocr.ocr(gray_plate, cls=True)
                if ocr_result:
                    for res in ocr_result:
                        if res:
                            for line in res:
                                detected_texts.add(line[1][0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        out.write(frame)

    cap.release()
    out.release()

    processed_video_path = os.path.join(PROCESSED_FOLDER, f"processed_{uuid.uuid4().hex}.mp4")
    convert_video_for_web(temp_video_path, processed_video_path)
    os.remove(temp_video_path)  # Remove temp video

    return processed_video_path, list(detected_texts)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                processed_file, ocr_results = process_video(file_path)
            else:
                processed_file, ocr_results = process_image(file_path)
            return render_template("result.html", uploaded_file=filename, processed_file=os.path.basename(processed_file), ocr_results=ocr_results)
    return render_template("index.html")


    
