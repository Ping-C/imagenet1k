python train_imagenet.py --config-file configs/simplevit/vits_1000cls_cml.yaml --training.altnorm=1 --lr.warmup_epochs=8 --logging.project_name imgnet1K_simplevits --logging.resume_id=aaa --logging.folder outputs/simplevits/baseline
python train_imagenet.py --config-file configs/simplevit/vits_1000cls_cml.yaml --training.randaug_version=v4 --model.arch=vit_s_v8 --training.altnorm=1 --lr.warmup_epochs=8 --logging.project_name imgnet1K_simplevits --logging.resume_id=aab --logging.folder outputs/simplevits/baseline_v2