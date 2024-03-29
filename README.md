# PatchComplete
This is the official code for "PatchComplete: Learning Multi-Resolution Patch Priors for 3D Shape Completion on Unseen Categories".

# Build environment for conda
```
conda env create -f patch_complete.yaml
conda activate patch_complete
```

# Download Processed Datasets

1. ScanNet
```
wget kaldir.vc.in.tum.de/yrao/scannet.zip
```

2. ShapeNet 
```
wget kaldir.vc.in.tum.de/yrao/shapenet.zip
```

3. Save datasets under `/data`.


# Data pre-processing for ShapeNet

Please follow the processing steps here: https://github.com/yinyunie/depth_renderer

# Data pre-processing for ScanNet

0. Download sdfs for ScanNet Scene
```
wget kaldir.vc.in.tum.de/yrao/scannet_2cm_sdf.zip
```

1. Get object GT info based on Scan2CAD annotation and ShapeNet models
```
python data_processing/gt_info_generation.py --annotation_file your/path/to/scan2cad_download_link/full_annotations.json --map_file your/path/to/scannetv2-labels.combined.csv --shapeNet_path /your/path/to/ShapeNet/ShapeNetCore.v2 --bbox_mesh_file your/path/to/Scan2CAD/Routines/Script/bbox.ply --output_path data_samples/scannet
```

2. Extract SDF inputs from ScanNet scenes
```
python data_processing/generate_sdfs_from_scannet_scenes.py --sdf_path your/path/to/scannet/scannet_2cm_sdf --scan_path your/path/to/ScanNet/public/v2/scans --mask_path dataset_samples/scannet
``` 

3. Generate scaled meshes for ShapeNet GT models based on Scan2CAD annotation
```
python data_processing/generate_scaled_meshes.py --scene_path your/path/to/ScanNet/public/v2/scans --gt_info_path data_samples/scannet --output_path data_samples/scannet
```

4. Generate TSDFs for scaled shapeNet meshes

Since we keep the scales for objects, which means we cannt simply normalize the ShapeNet models and move it to origin. 
externel/depth_fusion.py and externel/pyfusion are the modified version in this case. You can run it along with the code in pre-processing for ShapeNet.

# Training

0. Download Trained Models (rerun the experiments so have different numbers compared with the paper)
```
wget kaldir.vc.in.tum.de/yrao/trained_models.zip
```

1. Patch_learning on Shapenet dataset
```
python training.py --data_path data/shapenet --train_file txt_files/shapenet_train.txt --val_trained_file txt_files/shapenet_val.txt --val_novel_file txt_files/shapenet_val_novel.txt --truncation 2.5 --patch_res 32 --dataset shapenet --model_stage patch_learning
```
use `--no_wall_aug` for normal training (not for scannet pretrain)

2. Multires on Shapenet dataset
```
python training.py --data_path data/shapenet --train_file txt_files/shapenet_train.txt --val_trained_file txt_files/shapenet_val.txt --val_novel_file txt_files/shapenet_val_novel.txt --truncation 2.5 --dataset shapenet --model_stage multi_res
```
use `--no_wall_aug` for normal training (not for scannet pretrain)

3. Fine_tune on Scannet dataset
```
python training.py --data_path data/scannet --train_file txt_files/scannet_train.txt --val_trained_file txt_files/scannet_val.txt --val_novel_file txt_files/scannet_val_novel.txt --truncation 3 --dataset scannet --model_stage fine_tune --no_wall_aug
```

# Generation

1. Shapenet
```
python generation.py --data_path data/shapenet --model_name multi_res.pt --dataset shapenet --model_stage multi_res --truncation 2.5 --test_file txt_files/shapenet_test.txt
```
2. Scannet
```
python generation.py --data_path data/scannet --model_name fine_tune.pt --dataset scannet --model_stage multi_res --truncation 3 --test_file txt_files/scannet_test.txt
```

# Evaluation
```
cd evaluation
```
1. Shapenet
```
python evaluation.py --dataset shapenet --test_file ../txt_files/shapenet_test.txt --pred_path ../output --root ../data
```
2. Scannet
```
python evaluation.py --dataset scannet --test_file ../txt_files/scannet_test.txt --pred_path ../output --root ../data
```
