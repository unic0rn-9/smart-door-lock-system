import os
import cv2
import pickle
from deepface import DeepFace

# ✅ Path to your dataset folder
faces_folder = r'd:\Coding\Python Video\Python\Python Codes\IOT project\faces'

# ✅ Check if folder exists
if not os.path.exists(faces_folder):
    print(f"[ERROR] Folder not found: {faces_folder}")
    exit()

known_encodings = []
known_names = []

# ✅ Loop through each person's folder
for person_name in os.listdir(faces_folder):
    person_path = os.path.join(faces_folder, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"[INFO] Processing images for: {person_name}")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        print(f"  → Reading {img_name}")

        if not os.path.isfile(img_path):
            continue

        try:
            # Get the embedding from DeepFace
            embedding = DeepFace.represent(img_path=img_path, model_name='Facenet', enforce_detection=False)
            if embedding and len(embedding) > 0:
                known_encodings.append(embedding[0]['embedding'])
                known_names.append(person_name)
                print(f"    ✅ Encoded {img_name}")
            else:
                print(f"    ⚠️ No face detected in {img_name}")
        except Exception as e:
            print(f"    ❌ Error processing {img_name}: {e}")

# ✅ Save encodings to file
if len(known_encodings) > 0:
    data = {"encodings": known_encodings, "names": known_names}
    with open("face_encodings.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"\n[INFO] ✅ Encoding complete. Saved {len(known_encodings)} faces to face_encodings.pkl")
else:
    print("\n[WARNING] No encodings were created. Check your image folder or DeepFace installation.")
