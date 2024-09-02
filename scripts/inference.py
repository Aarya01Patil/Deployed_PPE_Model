import logging
import subprocess
import cv2
import torch
from ultralytics import YOLO
import os
from functools import lru_cache
import numpy as np
import tempfile

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@lru_cache(maxsize=None)
def load_person_model():
    return YOLO('weights/person_detection/best.pt').to(device)

@lru_cache(maxsize=None)
def load_ppe_model():
    return YOLO('weights/ppe_detection/best.pt').to(device)

def unload_models():
    load_person_model.cache_clear()
    load_ppe_model.cache_clear()
    torch.cuda.empty_cache()

def draw_boxes(img, boxes, labels, scores):
    class_colors = {
        "person": (0, 0, 255),
        "hard-hat": (0, 255, 0),
        "gloves": (0, 255, 255),
        "mask": (255, 255, 0),
        "glasses": (255, 0, 255),
        "boots": (255, 165, 0),
        "vest": (255, 182, 193),  
        "ppe-suit": (148, 0, 211),
        "ear-protector": (0, 128, 128),
        "safety-harness": (128, 128, 128),
    }
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        color = class_colors.get(label, (255, 255, 255))  
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        label_text = f"{label}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img, label_text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)

def process_frame(frame):
    person_model = load_person_model()
    ppe_model = load_ppe_model()

    person_results = person_model(frame)[0]

    all_boxes = []
    all_labels = []
    all_scores = []

    for person_box in person_results.boxes:
        x1, y1, x2, y2 = map(int, person_box.xyxy[0])
        person_score = person_box.conf.item()

        all_boxes.append((x1, y1, x2, y2))
        all_labels.append("person")
        all_scores.append(person_score)

        person_crop = frame[y1:y2, x1:x2]

        ppe_results = ppe_model(person_crop)[0]

        for ppe_box in ppe_results.boxes:
            px1, py1, px2, py2 = map(int, ppe_box.xyxy[0])
            ppe_class = ppe_results.names[int(ppe_box.cls.item())]
            ppe_score = ppe_box.conf.item()

            fx1, fy1 = x1 + px1, y1 + py1
            fx2, fy2 = x1 + px2, y1 + py2

            all_boxes.append((fx1, fy1, fx2, fy2))
            all_labels.append(ppe_class)
            all_scores.append(ppe_score)

    draw_boxes(frame, all_boxes, all_labels, all_scores)

    return frame

def process_image(input_file, output_file):
    image = cv2.imdecode(np.frombuffer(input_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError("Failed to load the image or image is empty")
    logging.info(f"Input image shape: {image.shape}")

    processed_image = process_frame(image)

    _, buffer = cv2.imencode('.jpg', processed_image)
    output_file.write(buffer)
    logging.info(f"Processed image saved to output file")

def process_video(input_file, output_file):
    try:
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input.write(input_file.read())
        temp_input.close()

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_output.close()

        video = cv2.VideoCapture(temp_input.name)
        if not video.isOpened():
            raise ValueError(f"Unable to open video file")

        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"Video properties: {width}x{height} at {fps} fps, {total_frames} frames")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"Unable to create output video file")

        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 1 != 0:  # Process every other frame
                continue
            
            processed_frame = process_frame(frame)
            out.write(processed_frame)

            if frame_count % 50 == 0:
                logging.info(f"Processed frame {frame_count}/{total_frames}")

        video.release()
        out.release()

        if frame_count == 0:
            raise ValueError("No frames were processed from the input video")

        ffmpeg_command = [
            'ffmpeg',
            '-i', temp_output.name,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-y',
            temp_output.name + '_final.mp4'
        ]
        
        try:
            subprocess.run(ffmpeg_command, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg error: {e.stderr.decode()}")
            raise
        except FileNotFoundError:
            logging.error("FFmpeg not found. Please ensure it's installed and in the system PATH.")
            raise

        with open(temp_output.name + '_final.mp4', 'rb') as final_video:
            output_file.write(final_video.read())

        os.unlink(temp_input.name)
        os.unlink(temp_output.name)
        os.unlink(temp_output.name + '_final.mp4')

        logging.info(f"Processed video saved to output file")
        logging.info(f"Processed {frame_count} frames")

    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        raise

def perform_inference(input_file, output_filename, output_file):
    if output_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        process_image(input_file, output_file)
    elif output_filename.lower().endswith(('.mp4', '.avi', '.mov')):
        process_video(input_file, output_file)
    else:
        raise ValueError("Unsupported file format")
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python inference.py <input_path> <output_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    perform_inference(input_path, output_path)