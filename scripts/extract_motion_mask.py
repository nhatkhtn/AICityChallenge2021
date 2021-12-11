from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from skimage.measure import label

alpha = 0.01
diff = 99
M = 13000

def get_motion_mask(video_path):
    cap = cv2.VideoCapture(str(video_path))
    
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
    tqdm.write(f"FPS = {cap.get(cv2.CAP_PROP_FPS)}")

    gap = fps // 5

    last_frame = None
    out = 0

    for idx in tqdm(range(total_frame_count), leave=False):
        # cap.set(cv2.CAP_PROP_POS_MSEC, 0.2 * 1000 * idx)
        ret, frame = cap.read()

        if not ret:
            break

        if idx % gap == 0:
            if idx % (gap * 5) == 0:
                last_frame = frame

            diff_frame = cv2.subtract(frame, last_frame)
            diff_frame = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
            _, diff_mask = cv2.threshold(diff_frame, 100, 1, cv2.THRESH_BINARY)

            if np.sum(diff_mask) <= M:
                out = cv2.bitwise_or(out, diff_frame)
                out = cv2.medianBlur(out, 3)
                out = cv2.GaussianBlur(out, (3, 3), 9)
                _, out = cv2.threshold(out, 99, 255, cv2.THRESH_BINARY)

    min_area = 10000
    mask = label(out, connectivity=1)
    num = np.max(mask)
    for i in range(1, int(num + 1)):
        if np.sum(mask==i) < min_area:
            mask[mask==i] = 0
    
    mask = np.expand_dims((mask > 0).astype(float), axis=-1)
    return mask

data_path = "/mnt/datah/Datasets/AIC21_track4/aic21-track4-test-data"
output_path = Path("output/motion/test")

output_path.mkdir(parents=True, exist_ok=True)

for vid_id in tqdm(range(1, 151)):
    video_path = Path(data_path) / f"{vid_id}.mp4"
    mask = get_motion_mask(video_path)
    img_path = output_path / f"{vid_id:03d}.jpg"
    cv2.imwrite(str(img_path), mask * 255)
    