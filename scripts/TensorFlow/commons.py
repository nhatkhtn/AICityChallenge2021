from configparser import ConfigParser


def default_configs():
    config = ConfigParser()
    config.add_section("main")
    config.set("main", "dataset_original_root",
               r"H:\Datasets\VinBigDataChestXray\vinbigdata-chest-xray-abnormalities-detection")
    config.set("main", "dataset_512_root", r"H:\Datasets\VinBigDataChestXray\512px")
    config.set("main", "coco_root", r"H:\Datasets\VinBigDataChestXray\512px\512-nms-ratio")
    return config

