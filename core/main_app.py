import cv2
import pickle
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import time

# --- Configuration ---
VIDEO_SOURCE = 3  # OBS virtual cam or webcam
SPOTS_FILE = 'core/parking_spots.pkl'
YOLO_MODEL = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = .01
# ---------------------

# --- Status Constants ---
STATUS_VACANT = 0
STATUS_OCCUPIED = 1
STATUS_RESERVED = 2

# --- Colors (BGR) ---
COLOR_VACANT = (0, 255, 255)   # Yellow
COLOR_RESERVED = (0, 255, 0)   # Green
COLOR_OCCUPIED = (0, 0, 255)   # Red
COLOR_DETECTION = (255, 0, 0)
# ---------------------

# --- Load YOLO ---
try:
    print("Loading YOLO model...")
    model = YOLO(YOLO_MODEL)
    VEHICLE_CLASS_IDS = [2, 5, 7]  # Car, bus, truck
    print("YOLO loaded.")
except Exception as e:
    print(f"Could not load YOLO: {e}")
    exit()

# --- Load Spots ---
try:
    with open(SPOTS_FILE, 'rb') as f:
        parking_spots = pickle.load(f)
        spot_detection_start = [None] * len(parking_spots)  # track when detection starts
except FileNotFoundError:
    print(f"'{SPOTS_FILE}' not found. Run spot_selector.py first.")
    exit()

spot_statuses = [STATUS_VACANT] * len(parking_spots)

# --- Init Video ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Cannot open source {VIDEO_SOURCE}")
    exit()

# Force 720p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# --- GUI App ---
class ParkingApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Parking Monitor")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Video panel
        self.monitor_label = tk.Label(self.root, bg="black")
        self.monitor_label.pack(fill="both", expand=True)

        # Control Panel (Touchscreen Interface)
        self.control_window = tk.Toplevel(self.root)
        self.control_window.title("Parking Kiosk")
        self.control_window.geometry("400x400")

        tk.Label(self.control_window, text="Parking Kiosk Interface",
                 font=("Helvetica", 16)).pack(pady=10)

        # --- Spot Input for Leaving ---
        tk.Label(self.control_window, text="Enter Spot Number to Leave:",
                 font=("Helvetica", 12)).pack(pady=5)
        self.card_entry = tk.Entry(self.control_window, font=("Helvetica", 14), justify="center")
        self.card_entry.pack(pady=5)

        # --- Buttons ---
        reserve_btn = tk.Button(self.control_window, text="Reserve Spot",
                                font=("Helvetica", 14), bg="#4CAF50", fg="white",
                                width=15, command=self.reserve_spot)
        reserve_btn.pack(pady=10)

        leave_btn = tk.Button(self.control_window, text="Leave Parking",
                              font=("Helvetica", 14), bg="#f44336", fg="white",
                              width=15, command=self.leave_spot)
        leave_btn.pack(pady=10)

        self.status_label = tk.Label(self.control_window, text="",
                                     font=("Helvetica", 12), fg="blue")
        self.status_label.pack(pady=10)

        self.update_frame()

    def reserve_spot(self):
        """Automatically assign a random vacant spot."""
        import random
        vacant_spots = [i for i, s in enumerate(spot_statuses) if s == STATUS_VACANT]

        if not vacant_spots:
            messagebox.showwarning("Full", "No vacant spots available.")
            return

        chosen_spot = random.choice(vacant_spots)
        spot_statuses[chosen_spot] = STATUS_RESERVED

        messagebox.showinfo("Spot Reserved",
                            f"Welcome! You have been assigned Spot #{chosen_spot + 1}.\nPlease proceed.")
        self.status_label.config(text=f"Spot #{chosen_spot + 1} Reserved.")
        self.card_entry.delete(0, tk.END)

    def leave_spot(self):
        """Simulate a car leaving by entering its spot number, with YOLO occupancy validation."""
        spot_id = self.card_entry.get().strip()
        if not spot_id.isdigit():
            messagebox.showwarning("Invalid Input", "Please enter a valid spot number.")
            return

        idx = int(spot_id) - 1
        if not (0 <= idx < len(spot_statuses)):
            messagebox.showwarning("Out of Range", "Invalid spot number.")
            self.card_entry.delete(0, tk.END)
            return

        # --- Run a quick YOLO frame check to verify car presence ---
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Camera Error", "Cannot capture video frame.")
            return

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        # Detect overlap between YOLO detections and the chosen parking spot
        spot_poly = parking_spots[idx]
        still_occupied = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in [2, 5, 7]:  # Car, bus, truck
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

                det_poly = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ]).reshape((-1, 1, 2))

                try:
                    overlap = cv2.intersectConvexConvex(
                        det_poly.astype(np.float32),
                        spot_poly.astype(np.float32)
                    )
                    if overlap[0] > 500:  # overlap threshold
                        still_occupied = True
                        break
                except cv2.error:
                    continue

        # --- Determine outcome ---
        if still_occupied:
            messagebox.showwarning("Spot Still Occupied",
                           f"Cannot leave yet — vehicle still detected in Spot #{idx+1}.")
        else:
            if spot_statuses[idx] in (STATUS_OCCUPIED, STATUS_RESERVED):
                spot_statuses[idx] = STATUS_VACANT
                messagebox.showinfo("Thank You", "Thank you!")
            else:
            # It was already vacant (no reservation, no car)
                messagebox.showinfo("Thank You", "Thank you!")
        self.card_entry.delete(0, tk.END)



    def update_frame(self):
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed.")
            self.on_closing()
            return

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        detections = []

        # Reset all non-reserved spots to Vacant for re-check
        for i, status in enumerate(spot_statuses):
            if status == STATUS_OCCUPIED:
                spot_statuses[i] = STATUS_VACANT

        current_time = time.time()
        detected_spots = set()  # to track which spots have detections this frame

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in [2, 5, 7]:  # Car, bus, truck
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])
                detections.append({"box": (x1, y1, x2, y2), "conf": conf})

                det_poly = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ]).reshape((-1, 1, 2))

                for i, spot_poly in enumerate(parking_spots):
                    try:
                        overlap = cv2.intersectConvexConvex(
                            det_poly.astype(np.float32),
                            spot_poly.astype(np.float32)
                        )

                        if overlap[0] > 500:  # overlap area threshold
                            detected_spots.add(i)

                            # If the spot is reserved, check time-based transition
                            if spot_statuses[i] == STATUS_RESERVED:
                                if spot_detection_start[i] is None:
                                    spot_detection_start[i] = current_time  # start timer
                                elif current_time - spot_detection_start[i] >= 3:
                                    # Vehicle present for >= 3 seconds — mark occupied
                                    spot_statuses[i] = STATUS_OCCUPIED
                                    spot_detection_start[i] = None  # reset timer
                            else:
                                spot_detection_start[i] = None  # not reserved

                            # For vacant spots — instantly mark occupied
                            if spot_statuses[i] == STATUS_VACANT:
                                spot_statuses[i] = STATUS_OCCUPIED
                            break

                    except cv2.error:
                        continue

        # Reset timers for spots not currently detected
        for i in range(len(parking_spots)):
            if i not in detected_spots:
                spot_detection_start[i] = None

        # Draw overlays
        frame = self.draw_overlays(frame.copy(), detections)

        # Convert to Tkinter image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img))
        self.monitor_label.imgtk = img_tk
        self.monitor_label.configure(image=img_tk)

        self.root.after(200, self.update_frame)


    def draw_overlays(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_DETECTION, 2)

        for i, spot_poly in enumerate(parking_spots):
            status = spot_statuses[i]

            if status == STATUS_OCCUPIED:
                color, text = COLOR_OCCUPIED, "Occupied"
            elif status == STATUS_RESERVED:
                color, text = COLOR_RESERVED, "Reserved"
            else:
                color, text = COLOR_VACANT, "Vacant"

            cv2.polylines(frame, [spot_poly], True, color, 3)

            M = cv2.moments(spot_poly)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = spot_poly[0][0]

            cv2.putText(frame, f"#{i+1} {text}",
                        (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2)

        occ = sum(1 for s in spot_statuses if s == STATUS_OCCUPIED)
        total = len(spot_statuses)
        cv2.putText(frame, f"Occupancy: {occ}/{total}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (255, 255, 255), 3)

        return frame

    def on_closing(self):
        print("Closing app...")
        cap.release()
        self.root.destroy()




if __name__ == "__main__":
    root = tk.Tk()
    app = ParkingApp(root)
    root.mainloop()
