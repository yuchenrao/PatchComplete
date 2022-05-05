'''
Generate watertight mesh from depth scans.
author: ynie
date: Jan, 2020
'''
import sys
sys.path.append('.')
import os
from external import pyfusion
from data_config import shapenet_rendering_path, camera_setting_path, total_view_nums, object_root, processing_file
from tools.read_and_write import read_exr, read_txt, load_scaled_obj_file
import numpy as np
# import mcubes
from multiprocessing import Pool
from functools import partial
from tools.utils import dist_to_dep

voxel_res = 128
truncation_factor = 12
s = 0

def process_mesh(obj_path, view_ids):
    '''
    script for prepare watertigt mesh for training
    :param obj (str): object path
    :param view_ids (N-d list): which view ids would like to render (from 1 to total_view_nums).
    :param cam_Ks (N x 3 x 3): camera intrinsic parameters.
    :param cam_RTs (N x 3 x 3): camera extrinsic parameters.
    :return:
    '''
    scene_name = obj_path.split('/')[-2]
    obj_name = obj_path.split('/')[-1]
    cam_K_path = os.path.join(camera_setting_path, 'cam_K', scene_name, obj_name[:-4], 'cam_K.txt')
    if os.path.exists(cam_K_path) is False:
        return None
    cam_K = np.loadtxt(cam_K_path)
    cam_Ks = np.stack([cam_K] * total_view_nums, axis=0).astype(np.float32)
    cam_RT_dir = [os.path.join(camera_setting_path, 'cam_RT',  scene_name, obj_name[:-4], 'cam_RT_{0:03d}.txt'.format(view_id)) for view_id in view_ids]
    cam_RTs = read_txt(cam_RT_dir)

    '''Decide save path'''
    scene_name, mask = obj_path.split('/')[-2:]
    output_path = os.path.join('your/path/to/scennet', scene_name) # change path
    if not os.path.exists(output_path):
	    os.mkdir(output_path)
    output_file = os.path.join(output_path, mask[:-4] + '_sdf_gt.npy')

    if os.path.exists(output_file):
        return None

    mask_file = obj_path[:-16] + 'mask.npz'
    with np.load(mask_file, "rb") as data:
        voxel_size = data["voxel_size"]
        voxel_origin = data["voxel_origin"]

    '''Begin to process'''
    obj_dir = os.path.join(shapenet_rendering_path, scene_name, obj_name[:-4])
    dist_map_dir = [os.path.join(obj_dir, 'depth_{0:03d}.exr'.format(view_id)) for view_id in view_ids]

    dist_maps = read_exr(dist_map_dir)
    # depth_maps = np.float32(dist_to_dep(dist_maps, cam_Ks, erosion_size=2))
    depth_maps = np.float32(dist_to_dep(dist_maps, cam_Ks, offset=1, erosion_size=3, voxel_size=voxel_size))
    # depth_maps = np.float32(dist_to_dep(dist_maps, cam_Ks))

    cam_Rs = np.float32(cam_RTs[:, :, :-1])
    cam_Ts = np.float32(cam_RTs[:, :, -1])

    truncation = truncation_factor * voxel_size
    point_clouds = np.zeros((depth_maps.shape[0], depth_maps.shape[1], depth_maps.shape[2], 3))
    pt_maps = np.zeros(point_clouds.shape)
    views = pyfusion.PyViews(depth_maps, cam_Ks, cam_Rs, cam_Ts, np.float32(point_clouds), np.float32(pt_maps))
    tsdf = pyfusion.tsdf_gpu(views, voxel_res, voxel_res, voxel_res, voxel_size, voxel_origin[0], voxel_origin[1], voxel_origin[2], truncation, False)

    # rotate to the correct system
    tsdf = np.transpose(tsdf[0], [2, 1, 0])
    np.save(output_file, tsdf)
    print (output_file)


if __name__ == '__main__':
    '''generate watertight meshes by patch'''
    view_ids = range(1, total_view_nums + 1)
    all_objects = load_scaled_obj_file(object_root, processing_file)

    p = Pool(processes=10)
    p.map(partial(process_mesh, view_ids=view_ids), all_objects)
    p.close()
    p.join()