#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main.py

Single-process script:
1) ThreadedVideoCapture for async camera input.
2) facenet-pytorch (MTCNN + InceptionResnetV1) for face detection & embedding (forced 160x160).
3) TFLite + Mediapipe for gesture detection.
4) Actually calls a Hikvision "open door" curl command with digest auth. 3-second cooldown.

Author: ChatGPT
"""

import os
import logging
import queue
import threading
import time
import itertools
import configparser
import subprocess
from string import Template

import cv2
import numpy as np
import mediapipe as mp_mediapipe
import torch
import torchvision.transforms as transforms

from facenet_pytorch import MTCNN, InceptionResnetV1

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')


##############################################################################
# 1. CONFIG READER
##############################################################################
def load_config(config_path='config.ini'):
    config = configparser.ConfigParser()
    with open(config_path) as f:
        content = f.read()
    # Expand environment variables in config
    content = Template(content).substitute(os.environ)
    from io import StringIO
    config.read_file(StringIO(content))
    logging.info("Configuration loaded from %s", config_path)
    return config


##############################################################################
# 2. ASYNCHRONOUS CAPTURE
##############################################################################
class ThreadedVideoCapture:
    """
    Grabs frames in a background thread from a local camera or GStreamer pipeline.
    read_latest() returns only the newest frame, discarding older ones.
    """

    def __init__(self, source=0, use_gstreamer=False):
        self.use_gstreamer = use_gstreamer
        if use_gstreamer:
            logging.debug("Opening capture with GStreamer pipeline: %s", source)
            self.cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        else:
            logging.debug("Opening capture: %s", source)
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            logging.error(f"Failed to open capture source: {source}")

        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _capture_loop(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                logging.debug("Failed to read frame in capture loop; sleeping.")
                time.sleep(0.1)
                continue
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def read_latest(self):
        latest_frame = None
        while not self.frame_queue.empty():
            latest_frame = self.frame_queue.get()
        return latest_frame

    def stop(self):
        logging.debug("Stopping capture thread.")
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.cap.release()


##############################################################################
# 3. TFLITE GESTURE CLASSIFIER
##############################################################################
class KeyPointClassifier:
    """
    Loads and runs a TFLite model for gesture classification (21 keypoints => 42 floats).
    """

    def __init__(self, model_path='model/keypoint_classifier.tflite'):
        import tensorflow as tf
        logging.info("Initializing gesture classifier with model: %s", model_path)
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        import numpy as np
        inp_idx = self.input_details[0]['index']
        self.interpreter.set_tensor(inp_idx, np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()
        out_idx = self.output_details[0]['index']
        result = self.interpreter.get_tensor(out_idx)
        gesture_id = int(np.argmax(np.squeeze(result)))
        return gesture_id


##############################################################################
# 4. GESTURE DETECTION HELPERS
##############################################################################
def calc_landmark_list(image, hand_landmarks):
    h, w = image.shape[:2]
    points = []
    for lm in hand_landmarks.landmark:
        x = min(int(lm.x * w), w - 1)
        y = min(int(lm.y * h), h - 1)
        points.append([x, y])
    return points

def pre_process_landmark(landmarks):
    base_x, base_y = landmarks[0]
    for p in landmarks:
        p[0] -= base_x
        p[1] -= base_y
    flat = list(itertools.chain.from_iterable(landmarks))
    max_val = max(map(abs, flat)) or 1
    return [v / max_val for v in flat]

def detect_gesture(frame, gesture_classifier, mp_hands_obj, mp_drawing):
    import mediapipe as mp
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands_obj.process(img_rgb)
    gesture_id = None
    if results.multi_hand_landmarks:
        logging.debug("Mediapipe found a hand.")
        hand_landmarks = results.multi_hand_landmarks[0]
        points = calc_landmark_list(frame, hand_landmarks)
        processed = pre_process_landmark(points)
        gesture_id = gesture_classifier(processed)
        logging.debug(f"Gesture classifier returned: {gesture_id}")
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    else:
        logging.debug("No hand detected by Mediapipe in this frame.")
    return gesture_id


##############################################################################
# 5. GSTREAMER PIPELINE BUILDER
##############################################################################
def build_gstreamer_pipeline(rtsp_url):
    pipeline = (
        f"rtspsrc location={rtsp_url} latency=50 ! "
        "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
    )
    return pipeline


##############################################################################
# 6. FACENET-PYTORCH: LOAD + RECOGNITION
##############################################################################
def load_known_faces_facenet(known_faces_dir, mtcnn, resnet, device='cpu'):
    emb_list = []
    name_list = []

    if not os.path.exists(known_faces_dir):
        logging.warning("Directory %s does not exist. No known faces loaded.", known_faces_dir)
        return emb_list, name_list

    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(known_faces_dir, filename)
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                logging.warning("Failed to read %s", path)
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            boxes, probs = mtcnn.detect(img_rgb)
            if boxes is None or len(boxes) < 1:
                logging.warning("No face found in known image %s", filename)
                continue
            x1, y1, x2, y2 = boxes[0].astype(int)
            h, w = img_rgb.shape[:2]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)

            face_w = x2 - x1
            face_h = y2 - y1
            if face_w < 20 or face_h < 20:
                logging.warning("Skipping tiny face in known image %s", filename)
                continue

            face_crop = img_rgb[y1:y2, x1:x2]
            # Force 160x160
            face_crop = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_AREA)

            face_tensor = transforms.ToTensor()(face_crop).to(device).unsqueeze(0)

            with torch.no_grad():
                emb = resnet(face_tensor).cpu().numpy()[0]
            name = os.path.splitext(filename)[0]
            emb_list.append(emb)
            name_list.append(name)
            logging.info(f"Known face loaded: {name}")

    return emb_list, name_list

def recognize_faces_facenet(frame_rgb,
                            mtcnn,
                            resnet,
                            known_embs,
                            known_names,
                            device='cpu',
                            distance_threshold=0.9,
                            min_box_size=20):
    boxes, probs = mtcnn.detect(frame_rgb)
    if boxes is None:
        return [], [], []

    recognized_names = []
    final_boxes = []
    final_probs = []

    h, w = frame_rgb.shape[:2]

    for box, prob in zip(boxes, probs):
        if box is None or prob < 0.8:
            continue
        x1, y1, x2, y2 = box.astype(int)

        # Clip
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        face_w = x2 - x1
        face_h = y2 - y1
        if face_w < min_box_size or face_h < min_box_size:
            logging.debug(f"Skipping tiny face: {face_w}x{face_h}")
            continue

        face_crop = frame_rgb[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # Force 160x160
        face_crop = cv2.resize(face_crop, (160,160), interpolation=cv2.INTER_AREA)

        face_tensor = transforms.ToTensor()(face_crop).to(device).unsqueeze(0)
        with torch.no_grad():
            emb = resnet(face_tensor).cpu().numpy()[0]

        best_dist = 999.0
        best_name = "Unknown"
        for known_emb, known_name in zip(known_embs, known_names):
            dist = np.linalg.norm(emb - known_emb)
            if dist < best_dist:
                best_dist = dist
                best_name = known_name

        if best_dist < distance_threshold:
            recognized_names.append(best_name)
        else:
            recognized_names.append("Unknown")

        final_boxes.append((x1, y1, x2, y2))
        final_probs.append(prob)
        logging.debug(f"Face recognized: {recognized_names[-1]}, dist={best_dist:.2f}, prob={prob:.2f}")

    return final_boxes, recognized_names, final_probs


##############################################################################
# 7. DOOR OPEN FUNCTION (with 3s cooldown)
##############################################################################

last_door_open_time = 0.0

def open_door_hikvision():
    """
    Calls the Hikvision API using credentials and URL from environment variables.
    """
    username = os.environ.get('HIK_USER', 'admin')
    password = os.environ.get('HIK_PASS', 'password')
    door_url = os.environ.get('HIK_DOOR_URL', 'http://localhost/ISAPI/AccessControl/RemoteControl/door/1')
    logging.info("Opening door via Hikvision API with curl (digest auth).")
    cmd = [
        "curl", "-s", "-S",
        "--digest", "-u", f"{username}:{password}",
        "-X", "PUT",
        "-d", "<RemoteControlDoor><cmd>open</cmd></RemoteControlDoor>",
        door_url
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logging.debug(f"Door open command success. Output:\n{result.stdout}")
        else:
            logging.error(f"Door open command failed. Return code={result.returncode}, err={result.stderr}")
    except Exception as e:
        logging.error(f"Failed to run curl for door opening: {e}")


def try_open_door_cooldown():
    global last_door_open_time
    now = time.time()
    if now - last_door_open_time >= 5.0:
        open_door_hikvision()
        last_door_open_time = now
    else:
        logging.debug("Skipping open door, cooldown not reached.")


##############################################################################
# 8. MAIN PROGRAM
##############################################################################
def main():
    logging.info("Starting Single-Process Facenet + Mediapipe App")

    config = load_config('config.ini')

    cam_source = config.get('Camera', 'source', fallback='laptop')
    doorbell_url = config.get('Camera', 'doorbell_url', fallback='')
    use_gstream = False
    source_str = 0

    if cam_source == 'doorbell' and doorbell_url:
        pipeline = build_gstreamer_pipeline(doorbell_url)
        logging.info(f"Using GStreamer pipeline: {pipeline}")
        source_str = pipeline
        use_gstream = True
    else:
        logging.info("Using laptop camera")

    cap = ThreadedVideoCapture(source_str, use_gstreamer=use_gstream).start()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    logging.info("Initializing MTCNN and InceptionResnetV1 for face detection & embedding.")
    mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.6, 0.7, 0.7])
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    known_faces_dir = config.get('FaceRecognition', 'known_faces_dir', fallback='known_faces')
    known_embs, known_names = load_known_faces_facenet(known_faces_dir, mtcnn, resnet, device=device)

    gesture_model_path = config.get('Gesture', 'model_path', fallback='model/keypoint_classifier.tflite')
    gesture_classifier = KeyPointClassifier(model_path=gesture_model_path)

    mp_hands = mp_mediapipe.solutions.hands
    mp_drawing = mp_mediapipe.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    door_api_url = config.get('DoorControl', 'api_url', fallback='http://192.168.1.137/ISAPI/AccessControl/RemoteControl/door/1')

    # Frame resizing
    resize_width = config.getint('Camera', 'resize_width', fallback=640)
    resize_height = config.getint('Camera', 'resize_height', fallback=480)

    logging.info("Entering main loop (single-process)")
    last_time = time.time()
    fps_text = "FPS: --"

    while True:
        frame = cap.read_latest()
        if frame is None:
            time.sleep(0.01)
            continue

        # Resize for speed
        frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection + recognition
        face_boxes, face_names, face_probs = recognize_faces_facenet(
            frame_rgb, mtcnn, resnet,
            known_embs, known_names,
            device=device,
            distance_threshold=0.9,
            min_box_size=30
        )

        # Draw bounding boxes
        for (x1,y1,x2,y2), name, prob in zip(face_boxes, face_names, face_probs):
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{name} {prob:.2f}", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Gesture detection
        gesture_id = detect_gesture(frame, gesture_classifier, hands, mp_drawing)

        # Door logic: if recognized face + gesture=0 => open door with 3s cooldown
        recognized_any = ("Unknown" not in face_names) and (len(face_names) > 0)
        if recognized_any and gesture_id == 2:
            logging.info("Authorized face + gesture=0 => Attempting to open door.")
            try_open_door_cooldown()

        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - last_time)
        last_time = current_time
        fps_text = f"FPS: {fps:.2f}"

        (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        H, W = frame.shape[:2]
        cx = (W - tw) // 2
        cy = (H + th) // 2
        cv2.putText(frame, fps_text, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        cv2.imshow("Facenet + Mediapipe (Single Process)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("User pressed 'q', exiting.")
            break

    cap.stop()
    cv2.destroyAllWindows()
    logging.info("Application terminated.")


if __name__ == "__main__":
    main()