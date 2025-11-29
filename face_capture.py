import cv2
import os

# Ask for user name (folder name)
user_name = input("Enter your name: ")

# Create a folder for this user
folder_path = f"dataset/{user_name}"
os.makedirs(folder_path, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)

count = 0
print("Capturing images... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the live frame
    cv2.imshow("Capturing Face - Press 'q' to stop", frame)

    # Save every few frames
    if count % 10 == 0:
        img_path = os.path.join(folder_path, f"{user_name}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved: {img_path}")

    count += 1

    # Key handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  
        print("Capture stopped.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"Face data collection complete for {user_name}")
