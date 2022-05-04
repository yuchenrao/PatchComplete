# PatchLearning
This is the official code for "Learning Multi-Resolution Patch Priors for 3D Shape Completion on Unseen Categories"

# Build environment for conda
```
conda env create -f patch_learning.yaml
conda activate patch_learning
```

# Data pre-processing for ShapeNet

Please follow the processing steps here: 

# Data pre-processing for ScanNet

1. Get object GT info based on Scan2CAD annotation and ShapeNet models
```
python data_processing/gt_info_generation.py --annotation_file ~/research/scan2cad_download_link/full_annotations.json --map_file scannetv2-labels.combined.csv --shapeNet_path /mnt/login_canis/ShapeNet/ShapeNetCore.v2 --bbox_mesh_file ~/research/Scan2CAD/Routines/Script/bbox.ply
```

2. Extract SDF inputs from ScanNet scenes
```
python data_processing/generate_sdfs_from_scannet_scenes.py
```

3. Generate scaled meshes for ShapeNet GT models based on Scan2CAD annotation
```
python data_processing/generate_scaled_meshes.py --scene_path /mnt/login_canis/Datasets/ScanNet/public/v2/scans --gt_info_path /mnt/login_cluster/gimli/yrao/output_new_coords/ --output_path /mnt/login_cluster/gimli/yrao/output_new_coords/
```

4. Generate TSDFs for scaled shapeNet meshes
Since we keep the scales for objects, which means we cannt simply normalize the ShapeNet models and move it to origin. 
externel/depth_fusion.py and externel/pyfusion are the modified version in this case. You can run it along with the code in pre-processing for ShapeNet.

# Training

1. Patch_learning on Shapenet dataset
```
python training.py --data_path data_samples/shapenet --train_file shapenet.txt --val_trained_file shapenet.txt --val_novel_file shapenet.txt --truncation 2.5 --patch_res 32 --dataset shapenet --model_stage patch_learning
```

2. Multires on Shapenet dataset
```
python training.py --data_path data_samples/shapenet --train_file shapenet.txt --val_trained_file shapenet.txt --val_novel_file shapenet.txt --truncation 2.5 --dataset shapenet --model_stage multi_res
```

3. Fine_tune on Scannet dataset
```
python training.py --data_path data_samples/scannet --train_file scannet.txt --val_trained_file scannet.txt --val_novel_file scannet.txt --truncation 3 --dataset scannet --model_stage fine_tune
```