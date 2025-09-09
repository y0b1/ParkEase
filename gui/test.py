import cv2
import json
import os
import numpy as np

IMG_PATH = r"C:\Users\Yobi\ParkEase\gui\parkingtest.jpg"
JSON_OUT = "parking_rois.json"
WINDOW = "ROI Selector"

if not os.path.exists(IMG_PATH):
    print("File not found:", IMG_PATH)
    raise SystemExit

base = cv2.imread(IMG_PATH)
if base is None:
    print("Failed to load image (format/path issue):", IMG_PATH)
    raise SystemExit

spots: list[list[list[int]]] = []   # list of polygons; each polygon is [[x,y],...]
current: list[list[int]] = []

def draw_scene():
    """Render base image + saved spots + current polygon + instructions."""
    img = base.copy()

    # Draw saved polygons (filled with low opacity + outline + index)
    if spots:
        overlay = img.copy()
        for idx, poly in enumerate(spots):
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 0, 255))  # red fill
            cv2.polylines(img, [pts], True, (0, 0, 200), 2)
            # label at centroid
            cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
            cv2.putText(img, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # 30% opacity for filled polygons
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    # Draw current polygon points/lines
    if current:
        for i, (x, y) in enumerate(current):
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            if i > 0:
                cv2.line(img, tuple(current[i-1]), (x, y), (255, 0, 0), 2)

    # Instructions banner
    banner = (
        "LMB:add point  |  ENTER/SPACE:finish spot  |  Z:undo  |  C:clear current  "
        "|  R:reset all  |  S:save  |  Q/ESC:quit"
    )
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, banner, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img

def save_json(path=JSON_OUT):
    data = {"spots": spots}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(spots)} spot(s) to {path}")

def on_mouse(event, x, y, flags, param):
    global current
    if event == cv2.EVENT_LBUTTONDOWN:
        current.append([int(x), int(y)])

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW, on_mouse)

while True:
    cv2.imshow(WINDOW, draw_scene())
    # Use waitKeyEx for better key handling on Windows
    k = cv2.waitKeyEx(50)

    if k == -1:
        continue

    # Quit: q, Q, or ESC
    if k in (ord('q'), ord('Q'), 27):
        # Optional auto-save on exit:
        if spots:
            save_json(JSON_OUT)
        break

    # Finish current spot: ENTER (13) or SPACE (32)
    if k in (13, 32):
        if len(current) >= 3:
            spots.append(current.copy())
            print(f"Spot #{len(spots)-1} saved with {len(current)} points")
        else:
            print("Need at least 3 points to form a polygon.")
        current = []

    # Undo last point
    elif k in (ord('z'), ord('Z')):
        if current:
            current.pop()

    # Clear current polygon
    elif k in (ord('c'), ord('C')):
        current = []

    # Reset everything
    elif k in (ord('r'), ord('R')):
        spots = []
        current = []
        print("Reset all spots")

    # Save JSON anytime
    elif k in (ord('s'), ord('S')):
        save_json(JSON_OUT)

cv2.destroyAllWindows()
