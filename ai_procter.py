import os
# Reduce TensorFlow / Mediapipe log spam
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import time
import threading
import csv
from datetime import datetime

# =========================
# CONFIGURATION
# =========================

VIDEO_DEVICE_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Risk weights
WEIGHT_LOOKING_AWAY = 0.5
WEIGHT_EXTRA_PERSON = 3.0
WEIGHT_PHONE = 4.0
WEIGHT_NOISE = 1.0

# Thresholds
LOOKING_AWAY_DISTANCE = 0.25     # how far from center = "away"
NO_FACE_THRESHOLD = 5.0          # seconds
AUDIO_FS = 16000
AUDIO_DURATION = 0.5             # seconds per audio chunk
AUDIO_NOISE_THRESHOLD = 0.02     # adjust depending on your mic

# =========================
# AUDIO MONITOR
# =========================

class AudioMonitor:
    def __init__(self, fs=AUDIO_FS, duration=AUDIO_DURATION, threshold=AUDIO_NOISE_THRESHOLD):
        self.fs = fs
        self.duration = duration
        self.threshold = threshold
        self.noise_detected = False
        self.running = False
        self.lock = threading.Lock()

    def _measure_noise(self):
        try:
            recording = sd.rec(
                int(self.duration * self.fs),
                samplerate=self.fs,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            energy = np.linalg.norm(recording) / len(recording)
            with self.lock:
                self.noise_detected = energy > self.threshold
        except Exception as e:
            print(f"[AudioMonitor] Error capturing audio: {e}")
            with self.lock:
                self.noise_detected = False

    def _loop(self):
        while self.running:
            self._measure_noise()
            time.sleep(0.1)

    def start(self):
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self.running = False

    def is_noise_detected(self):
        with self.lock:
            return self.noise_detected

# =========================
# AI PROCTOR (VISION + AUDIO + LOGGING)
# =========================

class AIProctorFull:
    def __init__(self):
        print("[INFO] Initializing MediaPipe FaceDetection...")
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

        # Try to load YOLO for phone detection
        self.yolo_available = False
        self.yolo_model = None
        try:
            from ultralytics import YOLO
            print("[INFO] Loading YOLOv8 model (yolov8n.pt) for phone detection...")
            self.yolo_model = YOLO("yolov8n.pt")  # will download on first use
            self.yolo_available = True
            print("[INFO] YOLO model loaded successfully.")
        except Exception as e:
            print("[WARN] Could not load YOLO model. Phone detection disabled.")
            print(f"[WARN] Reason: {e}")

        # State
        self.last_face_time = time.time()
        self.last_looking_away_start = None
        self.total_looking_away_time = 0.0

        self.extra_person_events = 0
        self.phone_events = 0

        self.last_noise_true_time = None
        self.total_noise_time = 0.0

        self.start_time = time.time()

        # Logging
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"proctor_log_{session_id}.csv"
        self.log_file = open(self.log_filename, mode="w", newline="", encoding="utf-8")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow([
            "timestamp",
            "session_time_sec",
            "num_faces",
            "looking_away",
            "extra_person",
            "phone_detected",
            "noise_detected",
            "risk_score"
        ])
        self.last_log_time = 0.0
        print(f"[INFO] Logging events to {self.log_filename}")

    # ---------- Utility methods ----------

    def _detect_faces(self, frame_rgb):
        results = self.face_detection.process(frame_rgb)
        bboxes = []
        if results.detections:
            h, w, _ = frame_rgb.shape
            for det in results.detections:
                box = det.location_data.relative_bounding_box
                x1 = int(box.xmin * w)
                y1 = int(box.ymin * h)
                x2 = int((box.xmin + box.width) * w)
                y2 = int((box.ymin + box.height) * h)
                bboxes.append((x1, y1, x2, y2))
        return bboxes

    def _is_looking_away(self, main_face_bbox, frame_shape):
        if main_face_bbox is None:
            return True
        x1, y1, x2, y2 = main_face_bbox
        fx = (x1 + x2) / 2
        fy = (y1 + y2) / 2
        h, w, _ = frame_shape
        cx = w / 2
        cy = h / 2
        dx = abs(fx - cx) / w
        dy = abs(fy - cy) / h
        return dx > LOOKING_AWAY_DISTANCE or dy > LOOKING_AWAY_DISTANCE

    def _run_yolo_phone(self, frame_bgr):
        """
        Run YOLO (if available) and count cell phones.
        Returns num_phones, annotations.
        """
        if not self.yolo_available or self.yolo_model is None:
            return 0, []

        results = self.yolo_model.predict(
            source=frame_bgr,
            imgsz=640,
            conf=0.4,
            verbose=False
        )

        num_phones = 0
        annotations = []

        if not results:
            return num_phones, annotations

        res = results[0]
        boxes = res.boxes
        if boxes is None:
            return num_phones, annotations

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            label = res.names[cls_id]

            if label == "cell phone":
                num_phones += 1
                annotations.append((x1, y1, x2, y2, label, conf))

        return num_phones, annotations

    # ---------- Main update per frame ----------

    def update_state(self, frame_bgr, audio_noise_flag):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        face_bboxes = self._detect_faces(frame_rgb)
        num_faces = len(face_bboxes)

        events = {
            "no_face": False,
            "extra_person": False,
            "looking_away": False,
            "phone_detected": False,
            "noise_detected": False
        }

        now = time.time()
        h, w, _ = frame_bgr.shape

        # Face presence
        if num_faces > 0:
            self.last_face_time = now

        if now - self.last_face_time > NO_FACE_THRESHOLD:
            events["no_face"] = True

        # Extra person based on faces
        if num_faces > 1:
            events["extra_person"] = True
            self.extra_person_events += 1

        # Main face = largest bbox
        main_face_bbox = None
        if num_faces > 0:
            areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in face_bboxes]
            main_face_bbox = face_bboxes[int(np.argmax(areas))]

        # Looking away
        if self._is_looking_away(main_face_bbox, frame_bgr.shape):
            events["looking_away"] = True
            if self.last_looking_away_start is None:
                self.last_looking_away_start = now
        else:
            if self.last_looking_away_start is not None:
                duration = now - self.last_looking_away_start
                self.total_looking_away_time += duration
                self.last_looking_away_start = None

        # Phone detection via YOLO
        num_phones, phone_annots = self._run_yolo_phone(frame_bgr)
        if num_phones > 0:
            events["phone_detected"] = True
            self.phone_events += num_phones

        # Audio / noise
        if audio_noise_flag:
            events["noise_detected"] = True
            if self.last_noise_true_time is None:
                self.last_noise_true_time = now
        else:
            if self.last_noise_true_time is not None:
                duration = now - self.last_noise_true_time
                self.total_noise_time += duration
                self.last_noise_true_time = None

        # Effective times (including currently active periods)
        effective_looking_away = self.total_looking_away_time
        if self.last_looking_away_start is not None:
            effective_looking_away += (now - self.last_looking_away_start)

        effective_noise_time = self.total_noise_time
        if self.last_noise_true_time is not None:
            effective_noise_time += (now - self.last_noise_true_time)

        # Risk score
        risk_score = (
            WEIGHT_LOOKING_AWAY * effective_looking_away +
            WEIGHT_EXTRA_PERSON * self.extra_person_events +
            WEIGHT_PHONE * self.phone_events +
            WEIGHT_NOISE * effective_noise_time
        )

        # ---------- Logging (once per second) ----------
        session_time = now - self.start_time
        if now - self.last_log_time >= 1.0:
            timestamp = datetime.now().isoformat(timespec="seconds")
            self.log_writer.writerow([
                timestamp,
                f"{session_time:.1f}",
                num_faces,
                int(events["looking_away"]),
                int(events["extra_person"]),
                int(events["phone_detected"]),
                int(events["noise_detected"]),
                f"{risk_score:.2f}"
            ])
            self.log_file.flush()
            self.last_log_time = now

        # ---------- Draw Overlays ----------
        annotated = frame_bgr.copy()

        # Face boxes
        for (x1, y1, x2, y2) in face_bboxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Phone boxes
        for (x1, y1, x2, y2, label, conf) in phone_annots:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

        info_lines = [
            f"Faces: {num_faces}",
            f"Looking away: {events['looking_away']}",
            f"Extra person: {events['extra_person']}",
            f"Phone detected: {events['phone_detected']}",
            f"Noise detected: {events['noise_detected']}",
            f"Looking-away time: {effective_looking_away:.1f}s",
            f"Noise time: {effective_noise_time:.1f}s",
            f"Phone events: {self.phone_events}",
            f"Extra person events: {self.extra_person_events}",
            f"Risk score: {risk_score:.2f}",
            f"Session: {session_time:.1f}s"
        ]

        y0 = 25
        dy = 22
        for i, line in enumerate(info_lines):
            cv2.putText(
                annotated,
                line,
                (10, y0 + i * dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        if risk_score > 15:
            cv2.putText(
                annotated,
                "HIGH RISK OF CHEATING",
                (int(w * 0.15), int(h * 0.9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

        return annotated, events, risk_score

    def close(self):
        try:
            self.log_file.close()
            print(f"[INFO] Log saved to {self.log_filename}")
        except Exception:
            pass

# =========================
# MAIN
# =========================

def main():
    print("[INFO] Opening webcam...")
    cap = cv2.VideoCapture(VIDEO_DEVICE_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Try changing VIDEO_DEVICE_INDEX.")
        return

    proctor = AIProctorFull()
    audio_monitor = AudioMonitor()
    audio_monitor.start()

    print("[INFO] Full AI Proctor running. Press 'q' on the video window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from webcam.")
                break

            noise_flag = audio_monitor.is_noise_detected()
            annotated, events, risk = proctor.update_state(frame, noise_flag)

            cv2.imshow("AI Proctor (Full)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] 'q' pressed, exiting.")
                break

    finally:
        audio_monitor.stop()
        cap.release()
        cv2.destroyAllWindows()
        proctor.close()
        print("[INFO] Proctor stopped cleanly.")

if __name__ == "__main__":
    main()
