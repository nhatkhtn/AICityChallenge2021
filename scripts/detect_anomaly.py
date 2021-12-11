from pathlib import Path

import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np

from commons import iou, area

detection_path = Path("output/raw/")
test_folder = Path("/mnt/datah/Datasets/AIC21_track4/aic21-track4-test-data")

min_detection_confidence = 0.3
event_idle_time_max = 10 * 5        # seconds
event_iou_continue_threshold = 0.3
event_anomaly_min_time = 30 * 5     # seconds
event_anomaly_min_conf = 0.65
event_min_end_time = 20 * 5

def get_fps(video_id):
    video_path = test_folder / f"{video_id}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps

def check_event(event):
    if event["last_update"] <= event_min_end_time:
        return False

    if event["last_update"] - event["start_time"] <= event_anomaly_min_time:
        return False

    if event["boxes"][-1][0] < 5 and area(event["boxes"][-1]) < 15 * 15:
        print(f"Small box killed")
        return False

    if np.mean(event["scores"]) < event_anomaly_min_conf:
        return False

    return True

soonest = {x : 123456789 for x in range(1, 151)}
conf = {x : 123456789 for x in range(1, 151)}

for video_id in range(1, 151):
    detection_csv = detection_path / f"{video_id}.csv"

    if not detection_csv.exists():
        print(f"csv for video {video_id} not found!")
        continue

    fps = get_fps(video_id)
    gap = int(fps + 0.5) // 5

    df = pd.read_csv(detection_csv)
    df = df[df["score"] >= min_detection_confidence]

    events = []
    anomalies = []

    for frame_id, table in df.groupby(["frame_id"]):
        time = int(frame_id) // gap

        new_candidates = []

        for x_min, y_min, x_max, y_max, score \
            in zip(table["x_min"], table["y_min"], table["x_max"], table["y_max"], table["score"]):
            
            box = (x_min, y_min, x_max, y_max)

            max_iou = -1
            matched_event = None

            for event in events:
                if time - event["last_update"] > event_idle_time_max:
                    event["stale"] = True
                    if check_event(event):
                        anomalies.append(event)
                else:
                    iou_event = iou(box, event["boxes"][-1])
                    if iou_event > event_iou_continue_threshold:
                        if iou_event > max_iou:
                            max_iou = iou_event
                            matched_event = event

            if matched_event is not None:
                matched_event["last_update"] = time
                matched_event["boxes"].append(box)
                matched_event["scores"].append(score)
            else:
                new_candidates.append((box, score))

        for box, score in new_candidates:
            events.append({
                "start_time": time,
                "last_update": time,
                "boxes": [box],
                "scores": [score],
                "stale": False,
            })

        events = [event for event in events if not event["stale"]]

    for event in events:
        if check_event(event):
            anomalies.append(event)

    if len(anomalies) > 0:
        for anomaly in anomalies:
            stime = int(anomaly["start_time"] * gap / fps)
            etime = int(anomaly["last_update"] * gap / fps)

            box = anomaly["boxes"][-1]
            avg_score = np.mean(anomaly["scores"])
            print(f"Video {video_id}, anomaly start {stime}, end {etime}, len = {etime-stime}, box: {box}, score: {avg_score}")
            if stime < soonest[video_id]:
                soonest[video_id] = stime
                conf[video_id] = avg_score
    else:
        print(f"Video {video_id}: no anomaly")

with open("submissions.txt", "w") as f:
    for i in range(1, 151):
        if soonest[i] != 123456789:
            print(i, soonest[i], conf[i], file=f)

    


