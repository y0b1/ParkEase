import cv2
import pickle
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# --- Configuration ---
VIDEO_SOURCE = 2  # OBS virtual cam or webcam
SPOTS_FILE = 'core/parking_spots.pkl'
YOLO_MODEL = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.1
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

        # Control Panel
        self.control_window = tk.Toplevel(self.root)
        self.control_window.title("Parking Controls")
        self.control_window.geometry("400x500")

        tk.Label(self.control_window, text="Parking Control Panel",
                 font=("Helvetica", 16)).pack(pady=10)

        reserve_btn = tk.Button(self.control_window, text="Reserve Spot",
                                font=("Helvetica", 14),
                                command=self.reserve_spot)
        reserve_btn.pack(pady=10)

        # Checkout buttons
        self.checkout_frame = tk.Frame(self.control_window)
        self.checkout_frame.pack(pady=10)

        self.checkout_buttons = []
        for i in range(len(parking_spots)):
            btn = tk.Button(self.checkout_frame,
                            text=f"Check Out Spot #{i+1}",
                            command=lambda idx=i: self.checkout_spot(idx))
            btn.grid(row=i // 2, column=i % 2, padx=5, pady=5)
            self.checkout_buttons.append(btn)

        self.update_frame()

    def reserve_spot(self):
        """Assign the first available vacant spot."""
        try:
            vacant_idx = spot_statuses.index(STATUS_VACANT)
            spot_statuses[vacant_idx] = STATUS_RESERVED
            messagebox.showinfo("Reserved",
                                f"Spot #{vacant_idx+1} reserved for you.")
        except ValueError:
            messagebox.showwarning("Full", "No vacant spots available.")

    def checkout_spot(self, idx):
        """Simulate car leaving."""
        if spot_statuses[idx] == STATUS_OCCUPIED or spot_statuses[idx] == STATUS_RESERVED:
            spot_statuses[idx] = STATUS_VACANT
            messagebox.showinfo("Checkout", f"Spot #{idx+1} is now Vacant.")
        else:
            messagebox.showwarning("Invalid", f"Spot #{idx+1} is already vacant.")

    def update_frame(self):
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed.")
            self.on_closing()
            return

        # Run YOLO
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        detections = []

        # Reset all non-reserved spots to Vacant for re-check
        for i, status in enumerate(spot_statuses):
            if status == STATUS_OCCUPIED:
                spot_statuses[i] = STATUS_VACANT

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in [2, 5, 7]:  # Car, bus, truck
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])
                detections.append({"box": (x1, y1, x2, y2), "conf": conf})

                # Build detection polygon
                det_poly = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ]).reshape((-1, 1, 2))

                for i, spot_poly in enumerate(parking_spots):
                    # Check polygon overlap
                    try:
                        overlap = cv2.intersectConvexConvex(
                            det_poly.astype(np.float32),
                            spot_poly.astype(np.float32)
                        )
                        if overlap[0] > 500:  # threshold in pixels²
                            if spot_statuses[i] != STATUS_RESERVED:
                                spot_statuses[i] = STATUS_OCCUPIED
                            break
                    except cv2.error:
                        continue

        # Draw overlays
        frame = self.draw_overlays(frame.copy(), detections)

        # Convert to Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img))
        self.monitor_label.imgtk = img_tk
        self.monitor_label.configure(image=img_tk)

        self.root.after(200, self.update_frame)

    def draw_overlays(self, frame, detections):
        # Debug: draw YOLO boxes
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_DETECTION, 2)

        # Draw spots
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

        # Occupancy counter
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
