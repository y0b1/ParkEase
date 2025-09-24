import os
import json
import cv2

# Paths
IMG_DIR = r"C:\Users\Yobi\Downloads\Car Park\images"
OUT_LABEL_DIR = r"C:\Users\Yobi\Downloads\Car Park\annotations"

os.makedirs(os.path.join(OUT_LABEL_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUT_LABEL_DIR, "val"), exist_ok=True)

# Map class names to IDs
CLASS_MAP = {
    "Occupied": 0,   # you can add more classes later if you have them
}

def convert_to_yolo(img_w, img_h, x1, y1, x2, y2):
    """Convert (x1,y1,x2,y2) to YOLO format (normalized)."""
    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return x_center, y_center, w, h

for split in ["train", "val"]:
    img_folder = os.path.join(IMG_DIR, split)
    label_folder = os.path.join(OUT_LABEL_DIR, split)

    for file in os.listdir(img_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(img_folder, file)
            json_path = img_path.rsplit(".", 1)[0] + ".json"
            label_path = os.path.join(label_folder, file.rsplit(".", 1)[0] + ".txt")

            if not os.path.exists(json_path):
                continue

            # Load image size
            img = cv2.imread(img_path)
            h, w, _ = img.shape

            with open(json_path, "r") as f:
                data = json.load(f)

            with open(label_path, "w") as out:
                for obj in data["labels"]:
                    cls_name = obj["name"]
                    if cls_name not in CLASS_MAP:
                        continue
                    cls_id = CLASS_MAP[cls_name]

                    x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]
                    x, y, bw, bh = convert_to_yolo(w, h, x1, y1, x2, y2)

                    out.write(f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

print("Conversion complete! YOLO labels saved.")

