from pathlib import Path

import pandas as pd

ver = "v1"

IMG_FOLDER = Path(r"H:\Workspace\AICity21\track4\data\samples\train\imgs")
OUTPUT_FOLDER = Path(r"H:\Workspace\AICity21\track4\data\samples\train")
ANNO_FILE = r"H:\Workspace\AICity21\track4\data\samples\train\annotations.csv"

VAL_RATIO = 0.1

if __name__ == '__main__':
    image_list = sorted(str(x).replace('\\\\', '\\') for x in IMG_FOLDER.iterdir())
    train_cnt = int(len(image_list) * (1 - VAL_RATIO))
    train_imgs = image_list[:train_cnt]
    val_imgs = image_list[train_cnt:]

    annos = pd.read_csv(ANNO_FILE, header=None)

    train_df = annos[annos.iloc[:, 0].isin(train_imgs)]
    train_df.to_csv(OUTPUT_FOLDER / f"train_{ver}.csv", header=None, index=None)

    val_df = annos[annos.iloc[:, 0].isin(val_imgs)]
    val_df.to_csv(OUTPUT_FOLDER / f"val_{ver}.csv", header=None, index=None)