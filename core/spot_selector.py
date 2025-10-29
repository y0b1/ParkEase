import cv2
import pickle
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import sv_ttk
from PIL import Image, ImageTk

# --- Configuration ---
VIDEO_SOURCE = 0  # Use 0 for webcam, or provide a path to a video file.
SPOTS_FILE = 'parking_spots.pkl'
FRAME_WIDTH = 1280  # 720p width
FRAME_HEIGHT = 720  # 720p height


# ---------------------

class SpotSelectorApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("ParkEase - Spot Selector")
        # 400px for controls + 1280px for video = 1680. 720px height.
        self.root.geometry("1680x720")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- Application State ---
        self.parking_spots = []
        self.current_spot_points = []

        # --- Init Video ---
        self.cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not self.cap.isOpened():
            print(f"Cannot open source {VIDEO_SOURCE}")
            # Show error in GUI instead of just console
            messagebox.showerror("Video Error", f"Cannot open video source {VIDEO_SOURCE}")
            self.root.destroy()
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # --- Setup the UI ---
        self.setup_ui()

        # --- Load existing spots ---
        self.load_spots()

        # --- Start the video loop ---
        print("UI Initialized. Starting video feed...")
        self.update_frame()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def setup_ui(self):
        """Orchestrates the UI creation."""
        self.setup_styles()
        self.setup_main_panels()
        self.setup_control_widgets()
        self.setup_bindings()

    def setup_styles(self):
        self.style = ttk.Style()
        font_family = "Segoe UI"

        self.style.configure("TLabel", font=(font_family, 11))
        self.style.configure("Heading.TLabel", font=(font_family, 20, "bold"))
        self.style.configure("TLabelframe.Label", font=(font_family, 12, "bold"))

        button_font = (font_family, 11, "bold")
        self.style.configure("TButton", font=button_font)
        self.style.configure("Accent.TButton", font=button_font)
        self.style.configure("Danger.TButton", font=button_font)

    def setup_main_panels(self):
        # --- Left Control Panel ---
        self.control_frame = ttk.Frame(self.root, width=400, padding=20)
        self.control_frame.pack(side="left", fill="y", expand=False)
        self.control_frame.pack_propagate(False)

        # --- Video Panel (Right Side) ---
        self.video_label = ttk.Label(self.root, background="black")  # Start with black bg
        self.video_label.pack(side="left", fill="both", expand=True)

    def setup_control_widgets(self):

        # --- Header ---
        header_label = ttk.Label(self.control_frame, text="Spot Selector",
                                 style="Heading.TLabel", anchor="center")
        header_label.pack(pady=(0, 10), fill="x")

        # --- Instructions Frame ---
        instr_frame = ttk.LabelFrame(self.control_frame, text="Instructions")
        instr_frame.pack(fill="x", pady=10)

        instr_text = "1. Left-click on the video feed to add points.\n" \
                     "2. Use the buttons below to manage spots."
        instr_label = ttk.Label(instr_frame, text=instr_text, justify="left")
        instr_label.pack(pady=10, padx=10, fill="x")

        # --- Spot Actions Frame ---
        actions_frame = ttk.LabelFrame(self.control_frame, text="Spot Actions")
        actions_frame.pack(fill="x", pady=10)

        # "Save Current Spot" (was 'n')
        self.btn_save_current = ttk.Button(actions_frame, text="Save Current Shape",
                                           style="Accent.TButton",
                                           command=self.save_current_spot)
        self.btn_save_current.pack(pady=(10, 5), padx=10, fill="x", ipady=10)

        # "Clear Current Drawing" (NEW helpful button)
        self.btn_clear_current = ttk.Button(actions_frame, text="Clear Current Drawing",
                                            style="Danger.TButton",
                                            command=self.clear_current_drawing)
        self.btn_clear_current.pack(pady=5, padx=10, fill="x", ipady=10)

        # "Remove Last Spot" (was right-click)
        self.btn_remove_last = ttk.Button(actions_frame, text="Remove Last Saved Spot",
                                          style="Danger.TButton",
                                          command=self.remove_last_spot)
        self.btn_remove_last.pack(pady=5, padx=10, fill="x", ipady=10)

        # --- File Actions Frame ---
        file_frame = ttk.LabelFrame(self.control_frame, text="File")
        file_frame.pack(fill="x", pady=10)

        # "Save to File" (was 's')
        self.btn_save_file = ttk.Button(file_frame, text="Save All Spots to File",
                                        style="Accent.TButton",
                                        command=self.save_spots_to_file)
        self.btn_save_file.pack(pady=(10, 15), padx=10, fill="x", ipady=10)

        # --- Status Label ---
        self.status_label = ttk.Label(self.control_frame, text="Loaded 0 spots.",
                                      style="Accent.TLabel", anchor="center")
        self.status_label.pack(side="bottom", pady=(10, 0), fill="x")

    def setup_bindings(self):
        """Binds mouse clicks to the video label."""
        # This replaces the cv2.setMouseCallback
        self.video_label.bind("<Button-1>", self.handle_video_click)

    # ------------------------------------------------------------------
    # Application Logic & Callbacks
    # ------------------------------------------------------------------

    def handle_video_click(self, event):
        self.current_spot_points.append((event.x, event.y))
        print(f"Added point: ({event.x}, {event.y})")
        # The update_frame loop will handle the redraw

    def save_current_spot(self):
        if len(self.current_spot_points) > 2:  # Need at least 3 points for a polygon
            self.parking_spots.append(np.array(self.current_spot_points, dtype=np.int32))
            self.current_spot_points = []
            self.status_label.config(text=f"Saved spot #{len(self.parking_spots)}.")
            print(f"Saved spot #{len(self.parking_spots)}.")
        else:
            self.status_label.config(text="Error: A spot needs at least 3 points.")
            print("A parking spot needs at least 3 points.")

    def clear_current_drawing(self):
        self.current_spot_points = []
        self.status_label.config(text="Current drawing cleared.")
        print("Current drawing cleared.")

    def remove_last_spot(self):
        if self.parking_spots:
            removed = self.parking_spots.pop()
            self.status_label.config(text=f"Removed last saved spot.")
            print("Removed the last saved parking spot.")
        else:
            self.status_label.config(text="No saved spots to remove.")
            print("No spots to remove.")

    def save_spots_to_file(self):
        with open(SPOTS_FILE, 'wb') as f:
            pickle.dump(self.parking_spots, f)
        self.status_label.config(text=f"Saved {len(self.parking_spots)} spots to '{SPOTS_FILE}'.")
        print(f"Successfully saved {len(self.parking_spots)} spots to '{SPOTS_FILE}'.")

    def load_spots(self):
        try:
            with open(SPOTS_FILE, 'rb') as f:
                self.parking_spots = pickle.load(f)
            self.status_label.config(text=f"Loaded {len(self.parking_spots)} spots from file.")
            print(f"Loaded {len(self.parking_spots)} spots.")
        except FileNotFoundError:
            self.status_label.config(text=f"'{SPOTS_FILE}' not found. Starting new file.")
            print(f"'{SPOTS_FILE}' not found. Starting new list.")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame. Check video source.")
            self.status_label.config(text="Error: Video feed lost.")
            return

        # Draw all overlays
        frame_with_overlays = self.draw_overlays(frame.copy())

        # Convert to Tkinter image
        img = cv2.cvtColor(frame_with_overlays, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img))

        # Update the label
        self.video_label.imgtk = img_tk
        self.video_label.configure(image=img_tk)

        # Schedule the next update
        self.root.after(30, self.update_frame)  # ~33 FPS

    def draw_overlays(self, frame):
        # Draw completed spots (Green)
        for spot in self.parking_spots:
            pts = spot.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw the points for the spot currently being defined (Red)
        for point in self.current_spot_points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

        # Draw lines for the current spot if more than one point exists
        if len(self.current_spot_points) > 1:
            for i in range(len(self.current_spot_points) - 1):
                cv2.line(frame, self.current_spot_points[i], self.current_spot_points[i + 1], (0, 0, 255), 2)

        return frame

    def on_closing(self):
        print("Closing application...")
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()

    sv_ttk.set_theme("light")

    app = SpotSelectorApp(root)
    root.mainloop()