import configparser
from pathlib import Path
from tqdm import tqdm

import cv2

config = configparser.ConfigParser()
config.read('config.ini')

dataset_path = Path(config['DEFAULT']['DatasetTrainPath'])

num_samples_per_video = 3

for video_path in tqdm(dataset_path.iterdir()):

    video_id = int(video_path.stem)

    cap = cv2.VideoCapture(str(video_path))
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = total_frame_count // num_samples_per_video

    for idx in range(0, total_frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            print("Read error!")
        else:
            img_name = f"{video_id}_{idx}.jpg"
            img_path = "data/samples/train/" + img_name
            cv2.imwrite(img_path, frame)
