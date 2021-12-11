from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

alpha = 0.02

data_path = "/mnt/datah/Datasets/AIC21_track4/aic21-track4-test-data"
output_path = Path("output/avg/test")

output_path.mkdir(parents=True, exist_ok=True)

for vid_id in [44]:
    video_path = Path(data_path) / f"{vid_id}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
    tqdm.write(f"FPS = {cap.get(cv2.CAP_PROP_FPS)}")

    gap = fps // 5
    
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    avg_frame = None

    Path(output_path / f"{vid_id:03d}").mkdir(exist_ok=True)

    for idx in tqdm(range(120 * 30)):
        ret, frame = cap.read()

        if not ret:
            break
        
        if idx % gap == 0:
            if avg_frame is None:
                avg_frame = frame
            else:
                avg_frame = (1 - alpha) * avg_frame + alpha * frame

            img_path = output_path / f"{vid_id:03d}" / f"{idx}.jpg"
            cv2.imwrite(str(img_path), avg_frame)
