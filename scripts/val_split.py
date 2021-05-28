import math
import itertools
import shutil
from pathlib import Path

import pandas as pd


def split_ratio(image_root=Path("/mnt/datah/Workspace/AICity21/track4/data/samples/train/imgs"),
                output_folder=Path("data/samples/train"),
                anno_file="data/samples/train/annotations_v2_annotated.csv",
                val_ratio=0.1):
    image_list = sorted(str(x).replace('\\\\', '\\') for x in image_root.iterdir())
    print(image_list[0])
    train_cnt = int(len(image_list) * (1 - val_ratio))
    train_imgs = image_list[:train_cnt]
    val_imgs = image_list[train_cnt:]

    annos = pd.read_csv(anno_file, header=None)

    split_list = list(itertools.zip_longest(train_imgs, [], fillvalue="train")) \
                 + list(itertools.zip_longest(val_imgs, [], fillvalue="val"))
    split_df = pd.DataFrame(split_list, columns=["images", "split"])
    split_df.to_csv(output_folder / "splits.csv", index=None)

    train_df = annos[annos.iloc[:, 0].isin(train_imgs)]
    train_df.to_csv(output_folder / f"train_{ver}.csv", header=None, index=None)

    val_df = annos[annos.iloc[:, 0].isin(val_imgs)]
    val_df.to_csv(output_folder / f"val_{ver}.csv", header=None, index=None)


def split_by_video(name_suffix, image_root=Path("data/samples/train/imgs"),
                   output_train=Path("data/samples/train/imgs_train"),
                   output_val=Path("data/samples/train/imgs_val"),
                   anno_file="data/samples/train/annotations.csv",
                   csv_output=Path("data/samples/train"),
                   use_existing_imgs=False):

    image_list = sorted(x.name for x in image_root.iterdir())

    if not use_existing_imgs:
        if not output_train.exists():
            output_train.mkdir()

        if not output_val.exists():
            output_val.mkdir()

        if list(output_train.iterdir()):
            print("WARNING! Train output folder is not empty.")

        if list(output_val.iterdir()):
            print("WARNING! Val output folder is not empty.")

        for video_id, group in itertools.groupby(image_list, key=lambda x : x.split('_')[0]):
            images = list(group)
            train_cnt = math.ceil(len(images) * 0.5)
            for image_name in images[:train_cnt]:
                shutil.copyfile(image_root / image_name, output_train / image_name)
            for image_name in images[train_cnt:]:
                shutil.copyfile(image_root / image_name, output_val / image_name)

    annos = pd.read_csv(anno_file, header=None)

    def create_df_from_images_folder(anno_df, folder):
        image_names = {image_path.name for image_path in folder.iterdir()}
        return anno_df[anno_df[0].isin(image_names)]

    train_df = create_df_from_images_folder(annos, output_train)
    val_df = create_df_from_images_folder(annos, output_val)

    train_df.to_csv(csv_output / f"train_{name_suffix}.csv", header=None, index=None)
    val_df.to_csv(csv_output / f"val_{name_suffix}.csv", header=None, index=None)


if __name__ == '__main__':
    split_by_video("v3", use_existing_imgs=True)
