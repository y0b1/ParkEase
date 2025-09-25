import cv2
import pickle
import numpy as np

# --- Configuration ---
VIDEO_SOURCE = 2  # Use 0 for webcam, or provide a path to a video file.
SPOTS_FILE = 'parking_spots.pkl'
FRAME_WIDTH = 1280  # 720p width
FRAME_HEIGHT = 720  # 720p height
# ---------------------

cap = cv2.VideoCapture(VIDEO_SOURCE)

# Force resolution to 720p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

try:
    with open(SPOTS_FILE, 'rb') as f:
        parking_spots = pickle.load(f)
except FileNotFoundError:
    parking_spots = []

current_spot_points = []

def mouse_callback(event, x, y, flags, params):
    """Handles mouse clicks to define polygon points for parking spots."""
    global current_spot_points

    if event == cv2.EVENT_LBUTTONDOWN:
        current_spot_points.append((x, y))

    if event == cv2.EVENT_RBUTTONDOWN:
        # Right-click to remove the last saved spot
        if parking_spots:
            parking_spots.pop()
            print("Removed the last saved parking spot.")

def draw_spots(frame):
    """Draws all defined spots and the currently drawn spot on the frame."""
    # Draw completed spots
    for spot in parking_spots:
        pts = np.array(spot, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw the points for the spot currently being defined
    for point in current_spot_points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)

    # Draw lines for the current spot if more than one point exists
    if len(current_spot_points) > 1:
        for i in range(len(current_spot_points) - 1):
            cv2.line(frame, current_spot_points[i], current_spot_points[i+1], (0, 0, 255), 2)


print("--- Parking Spot Selector ---")
print("Left-click to add points for a parking spot polygon.")
print("Press 'n' to save the current spot and start a new one.")
print("Right-click to remove the last saved spot.")
print("Press 's' to save all spots to file.")
print("Press 'q' to quit.")
print("-----------------------------")


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame. Check video source.")
        break

    cv2.namedWindow('Spot Selector')
    cv2.setMouseCallback('Spot Selector', mouse_callback)

    draw_spots(frame)

    cv2.imshow('Spot Selector', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('n'):  # 'n' for new spot
        if len(current_spot_points) > 2:  # A polygon needs at least 3 points
            parking_spots.append(np.array(current_spot_points, dtype=np.int32))
            print(f"Saved spot #{len(parking_spots)}.")
            current_spot_points = []
        else:
            print("A parking spot needs at least 3 points.")

    if key == ord('s'):  # 's' to save to file
        with open(SPOTS_FILE, 'wb') as f:
            pickle.dump(parking_spots, f)
        print(f"Successfully saved {len(parking_spots)} spots to '{SPOTS_FILE}'.")


cap.release()
cv2.destroyAllWindows()
