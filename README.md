# PatchLearning
This is the official code for "Learning Multi-Resolution Patch Priors for 3D Shape Completion on Unseen Categories"


# Data pre-processing for ScanNet

1. Get object GT info based on Scan2CAD annotation and ShapeNet models
```
python gt_info_generation.py --annotation_file ~/research/scan2cad_download_link/full_annotations.json --map_file scannetv2-labels.combined.csv --shapeNet_path /mnt/login_canis/ShapeNet/ShapeNetCore.v2 --bbox_mesh_file ~/research/Scan2CAD/Routines/Script/bbox.ply
```

2. Extract SDF inputs from ScanNet scenes
```
python generate_sdfs_from_scannet_scenes.py
```

3. Generate SDF for ShapeNet GT models
```

```