import cv2
import json
import numpy as np
import os

# --- CONFIG ---
ROI_FILE = "parking_rois.json"   # saved from your ROI selector
CAMERA_INDEX = 2            # 0 = default webcam, or use video path instead
THRESHOLD = 30                   # sensitivity (lower = stricter)

# --- LOAD ROIS ---
if not os.path.exists(ROI_FILE):
    print("No ROI file found! Run the ROI selector first.")
    exit()

with open(ROI_FILE, "r") as f:
    rois = json.load(f)["spots"]

print(f"Loaded {len(rois)} parking spots.")

# --- CAPTURE ---
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Cannot open camera/video")
    exit()

# --- BACKGROUND REFERENCE (first frame as 'empty lot') ---
ret, ref_frame = cap.read()
if not ret:
    print("Failed to grab reference frame")
    exit()

ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

def check_spot(frame_gray, roi):
    """Return True if occupied, False if vacant."""
    mask = np.zeros_like(frame_gray, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(roi, np.int32)], 255)

    # Extract ROI region
    spot_current = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)
    spot_ref = cv2.bitwise_and(ref_gray, ref_gray, mask=mask)

    # Compare difference
    diff = cv2.absdiff(spot_current, spot_ref)
    score = cv2.mean(diff, mask=mask)[0]  # average pixel diff

    return score > THRESHOLD, score  # occupied?, raw score

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    overlay = frame.copy()

    for idx, roi in enumerate(rois):
        occupied, score = check_spot(frame_gray, roi)

        color = (0, 0, 255) if occupied else (0, 255, 0)  # red=occupied, green=vacant
        pts = np.array(roi, np.int32)

        # Draw transparent polygon
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], True, color, 2)

        # Label
        cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        label = f"{'OCC' if occupied else 'FREE'} ({int(score)})"
        cv2.putText(frame, label, (cx-20, cy),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

    # Blend overlay with transparency
    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.imshow("Parking Availability", frame)
    key = cv2.waitKey(30)

    if key in (ord('q'), 27):  # quit on Q or ESC
        break
    elif key == ord('r'):  # reset reference frame
        ref_gray = frame_gray.copy()
        print("Reference frame reset!")

cap.release()
cv2.destroyAllWindows()
