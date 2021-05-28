Train command:

    python model_main_tf2.py --model_dir=models/my_faster_rcnn_v3/ --pipeline_config_path=models/my_faster_rcnn_v3/pipeline.config --alsologtostderr

Eval command:

    CUDA_VISIBLE_DEVICES=0 
    python model_main_tf2.py --model_dir=models/my_faster_rcnn_v3/ --pipeline_config_path=models/my_faster_rcnn_v3/pipeline.config --checkpoint_dir=models/my_faster_rcnn_v3


Export command:

    python exporter_main_v2.py --input_type image_tensor --pipeline_config_path models/my_efficientdet_v4/pipeline.config --trained_checkpoint_dir models/my_efficientdet_v4 --output_directory exported-models/my_efficientdet_v4 

