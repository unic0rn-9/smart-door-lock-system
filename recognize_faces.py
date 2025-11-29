# recognize_faces.py
import cv2
import pickle
import numpy as np
import time
import os
import math
from collections import deque, Counter
from deepface import DeepFace

# ---------------- CONFIG ----------------
ENCODINGS_FILE = "face_encodings.pkl"
TEMP_IMG = "temp_frame.jpg"
PROCESS_EVERY_N = 4        # run embedding every N frames
SMOOTH_WINDOW = 6         # number of recent results to keep for consensus
THRESHOLD = 0.55          # cosine similarity threshold (0.5-0.65 typical)
MIN_CONSENSUS_RATIO = 0.5 # fraction of window that must agree
RESIZE_WIDTH = 640        # speed-up frame width
# ----------------------------------------

# load encodings
if not os.path.exists(ENCODINGS_FILE):
    print(f"[ERROR] Encodings file not found: {ENCODINGS_FILE}")
    exit(1)

with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)

known_encodings = [np.array(e, dtype=np.float32) for e in data.get("encodings", [])]
known_names = data.get("names", [])

if len(known_encodings) == 0:
    print("[ERROR] No encodings found in the file.")
    exit(1)

# normalize known encodings (unit vectors) for cosine similarity
known_encodings = [v / np.linalg.norm(v) if np.linalg.norm(v) != 0 else v for v in known_encodings]
known_encodings = np.stack(known_encodings)  # shape (M, D)

print(f"[INFO] Loaded {len(known_encodings)} encoded faces: {set(known_names)}")
print("[INFO] Starting webcam... (press 'q' to quit)")

# face detector for drawing box (uses OpenCV haarcascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit(1)

frame_count = 0
recent_scores = deque(maxlen=SMOOTH_WINDOW)  # store best similarity scores
recent_names = deque(maxlen=SMOOTH_WINDOW)   # store best-matching name per processed frame
last_result_text = "Detecting..."
last_result_color = (255, 255, 0)
last_match_time = 0
DISPLAY_DURATION = 2.0  # seconds to show last result

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        h, w = frame.shape[:2]
        # resize a copy for speed (we'll draw on original frame)
        if w > RESIZE_WIDTH:
            scale = RESIZE_WIDTH / float(w)
            frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            frame_small = frame.copy()
            scale = 1.0

        frame_count += 1

        # run embedding only every N frames to reduce CPU load
        if frame_count % PROCESS_EVERY_N == 0:
            # write temp image (DeepFace prefers path input)
            cv2.imwrite(TEMP_IMG, frame_small)

            try:
                rep = DeepFace.represent(img_path=TEMP_IMG, model_name="Facenet", enforce_detection=False)
                if rep and len(rep) > 0:
                    emb = np.array(rep[0]["embedding"], dtype=np.float32)
                    # normalize embedding
                    nrm = np.linalg.norm(emb)
                    if nrm == 0:
                        normalized_emb = emb
                    else:
                        normalized_emb = emb / nrm

                    # cosine similarities with knowns (dot product since normalized)
                    sims = np.dot(known_encodings, normalized_emb)  # shape (M,)
                    best_idx = int(np.argmax(sims))
                    best_score = float(sims[best_idx])
                    best_name = known_names[best_idx]

                    # push into recent history
                    recent_scores.append(best_score)
                    recent_names.append(best_name)

                    # debug print
                    print(f"[DEBUG] frame#{frame_count} best={best_name} score={best_score:.4f}")

                else:
                    # no embedding produced (no face)
                    recent_scores.append(0.0)
                    recent_names.append("Unknown")
                    print(f"[DEBUG] frame#{frame_count} no embedding")

            except Exception as ex:
                recent_scores.append(0.0)
                recent_names.append("Unknown")
                print("[ERROR] DeepFace exception:", ex)

            # consensus decision:
            avg_score = float(np.mean(recent_scores)) if len(recent_scores) > 0 else 0.0
            most_common, count = Counter(recent_names).most_common(1)[0] if len(recent_names) > 0 else ("Unknown", 0)
            needed = math.ceil(len(recent_names) * MIN_CONSENSUS_RATIO) if len(recent_names) > 0 else 1

            if avg_score >= THRESHOLD and count >= needed and most_common != "Unknown":
                last_result_text = f"Access Granted ✅ Welcome, {most_common}"
                last_result_color = (0, 200, 0)
                last_match_time = time.time()
                # print a clear event when granted
                print(f"[ACCESS GRANTED] {most_common} (avg_score={avg_score:.3f}, votes={count}/{len(recent_names)})")
            elif len(recent_names) > 0 and most_common == "Unknown" and avg_score < THRESHOLD:
                last_result_text = "No face detected"
                last_result_color = (0, 255, 255)
            else:
                # deny if consensus not reached or avg too low
                last_result_text = "Access Denied ❌"
                last_result_color = (0, 0, 255)

        # draw bounding boxes using Haar (on frame_small to find faces)
        gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=5)

        # draw rectangles on original-size frame (rescale coordinates)
        for (x, y, wbox, hbox) in faces:
            x_orig = int(x / scale)
            y_orig = int(y / scale)
            w_orig = int(wbox / scale)
            h_orig = int(hbox / scale)
            cv2.rectangle(frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), (200, 200, 0), 2)

        # decide what to display (show last_result for a short time)
        if time.time() - last_match_time > DISPLAY_DURATION:
            display_text = "Detecting..."
            display_color = (255, 255, 0)
        else:
            display_text = last_result_text
            display_color = last_result_color

        # show similarity value if available
        score_text = ""
        if len(recent_scores) > 0:
            score_text = f"Score:{np.mean(recent_scores):.3f}"

        cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, display_color, 2)
        cv2.putText(frame, score_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Smart Door Lock - Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists(TEMP_IMG):
        try:
            os.remove(TEMP_IMG)
        except:
            pass
    print("[INFO] Exited.")
