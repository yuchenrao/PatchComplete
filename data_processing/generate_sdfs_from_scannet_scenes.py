import numpy as np
import struct
import IPython
import os
import glob
from scipy import interpolate
from plyfile import PlyData, PlyElement
import skimage.measure
import trimesh
import math
import time
import multiprocessing as mp
from multiprocessing import Pool
import gc

class SampleSDF:
    def __init__(self, dims=[0, 0, 0], res=0, world2grid=None, num_locations=None, locations=None, sdfs=None):
        self.filename = ""
        self.dimx = dims[0]
        self.dimy = dims[1]
        self.dimz = dims[2]
        self.res = res
        self.world2grid = world2grid
        self.num_locations = num_locations
        self.locations = locations
        self.sdfs = sdfs

class SampleKNW:
    def __init__(self, dims=[0, 0, 0], res=0, world2grid=None, knowns=None):
        self.filename = ""
        self.dimx = dims[0]
        self.dimy = dims[1]
        self.dimz = dims[2]
        self.res = res
        self.world2grid = world2grid
        self.knowns = knowns
        
class TsdfExtractor:
    def __init__(self, sdf_path="/mnt/login_cluster_HDD/sorona/adai/data/scannet/scannet_2cm_sdf",
                       scan_path="/mnt/login_canis/Datasets/ScanNet/public/v2/scans/",
                       mask_path="/mnt/login_cluster/gimli/yrao/output_new_coords",
                       grid_res=32):
        self._sdf_path = sdf_path
        self._scan_path = scan_path
        self._mask_path = mask_path
        self._scene_path = None
        self._truncated_value = 0.15
        self._debug = False
        # get bbox mesh
        # bbox_file = "/mnt/login_cluster/gimli/yrao/bbox.ply" # bbox model from Scan2CAD dataset
        # self._bbox_mesh = trimesh.load(bbox_file)
        # bbox padding
        self._voxel_extend = 2
        self._grid_res = grid_res
        self._bad_scene = []
        self._bad_cases = []

    def extract_data_single(self, scene_name):
        
        self._scene_path = os.path.join(self._mask_path, scene_name)

        # get sdf file
        sdf_filename = os.path.join(self._sdf_path, scene_name + ".sdf")
        knw_filename = os.path.join(self._sdf_path, scene_name + ".knw")
        trans_filename = os.path.join(os.path.join(self._scan_path, scene_name), scene_name + ".txt")
        if os.path.exists(sdf_filename) is not True:
            print ("No sdf found for " + scene_name)
            return

        print("processing: " + scene_name)
        with open(trans_filename) as reader:
            lines = reader.readlines()
        
        # read data
        l = lines[0].split(" ")
        if (len(l) == 19):
            T2axisalign = np.array([[float(l[2]), float(l[3]), float(l[4]), float(l[5])],
                                    [float(l[6]), float(l[7]), float(l[8]), float(l[9])],
                                    [float(l[10]), float(l[11]), float(l[12]), float(l[13])],
                                    [float(l[14]), float(l[15]), float(l[16]), float(l[17])]],
                                    dtype=np.float32)
        elif (len(l) == 3):
            l_new = lines[1].split(" ")
            if (len(l_new) == 19):
                T2axisalign = np.array([[float(l_new[2]), float(l_new[3]), float(l_new[4]), float(l_new[5])],
                                        [float(l_new[6]), float(l_new[7]), float(l_new[8]), float(l_new[9])],
                                        [float(l_new[10]), float(l_new[11]), float(l_new[12]), float(l_new[13])],
                                        [float(l_new[14]), float(l_new[15]), float(l_new[16]), float(l_new[17])]],
                                        dtype=np.float32)
            else:
                self._bad_scene.append(scene_name)
                print ("invalid trans file")
                return
        else:
            self._bad_scene.append(scene_name)
            print ("invalid trans file")
            return

        try:
            # get sdf sdf knw data
            sample_sdf = self.load_sample_sdf(sdf_filename)
            sample_knw = self.load_sample_knw(knw_filename)
            assert sample_sdf.dimx == sample_knw.dimx and sample_sdf.dimy == sample_knw.dimy and sample_sdf.dimz == sample_knw.dimz
            
            # get scene mesh info
            scene_mesh_file = os.path.join(os.path.join(self._scan_path, scene_name), scene_name + "_vh_clean_2.ply")
            scene_mesh = trimesh.load(scene_mesh_file, process=False)
            scene_mesh.apply_transform(T2axisalign)
            Tori = np.identity(4)
            Tori[:3, 3] = -1 * scene_mesh.bounds[0]
            Tscannet2generate = Tori @ T2axisalign
        except:
            self._bad_scene.append(scene_name)
            print ('failed on reading sdf data')
            return

        # init dense sdf
        sdf_vox = np.ones((sample_sdf.dimx, sample_sdf.dimy, sample_sdf.dimz)) * -1 * self._truncated_value # unseen
        sdf_vox[sample_knw.known < 2] = self._truncated_value # empty + surface
        sdf_vox[sample_sdf.locations[:, 0], sample_sdf.locations[:, 1], sample_sdf.locations[:, 2]] = sample_sdf.sdfs
        sdf_grid_coords = (np.array(range(0, sample_sdf.dimx)),
                            np.array(range(0, sample_sdf.dimy)),
                            np.array(range(0, sample_sdf.dimz)))

        # get obj files
        mask_idx = 0
        for mask_file in glob.iglob(os.path.join(self._scene_path, "*mask.npz")):
            instance_sdf_file = mask_file[:-8] + "_sdf.npz"
            # get sdf and mask for instance
            try:
                instance_sdf, mask_file = self.get_instance_data(mask_file, Tscannet2generate, sample_sdf, sdf_grid_coords, sdf_vox, mask_idx)
                # np.savez(instance_sdf_file, instance_sdf=instance_sdf, instance_mask=instance_mask)
                np.savez(instance_sdf_file, instance_sdf=instance_sdf)
            except:
                self._bad_cases.append(mask_file)
            mask_idx += 1
        print (gc.collect())

        return

    def extract_data_multi_process(self):
        _, scene_names, _ = next(os.walk(self._mask_path))
        rest_scenes = []
        for scene_name in scene_names:
            instance_list = []
            num = 0
            scene_path = os.path.join(self._mask_path, scene_name)
            for obj_file in glob.iglob(os.path.join(scene_path, "*mask.obj")):
                path_list = obj_file.split("/")[-1].split("_")
                mask_file = os.path.join(scene_path, path_list[1] + "_" + path_list[2] + "_" + path_list[3] + "_mask.npz")
                instance_sdf_file = mask_file[:-8] + "_sdf.npz"
                if os.path.exists(instance_sdf_file):
                    instance_list.append(instance_sdf_file)
                num += 1
            if (len(instance_list) < num):
                rest_scenes.append(scene_name)
        print (len(rest_scenes))
        p = Pool(processes=8)
        out = p.map(self.extract_data_single, rest_scenes)
        p.close()
        p.join()
        return out

    def get_instance_data(self, mask_file, Tscannet2generate, sample_sdf, sdf_grid_coords, sdf_vox, mask_idx):
        # print (mask_file)
        with np.load(mask_file) as data:
            Mmodel2scannet=data["Mmodel2scannet"]
            Mbbox2model = data["Mbbox2model"]
            voxel_size = data["voxel_size"]
            voxel_origin = data["voxel_origin"]
            Tscales = data["T_scales"]

        # get related transformation matrix
        Tmodel2generate = Tscannet2generate @ Mmodel2scannet
        Tmodel2sdfgrid = np.matrix.transpose(sample_sdf.world2grid) @ Tmodel2generate
        TshapeNetGrid2model = np.identity(4) * voxel_size
        TshapeNetGrid2model[3][3] = 1
        TshapeNetGrid2model[:3, 3] = voxel_origin
        TshapeNetGrid2sdfGrid = Tmodel2sdfgrid @ TshapeNetGrid2model
        # init sdf_final for this object and get points for 32 ^ 3 in shapeNet grid coords
        sdf_final = np.ones((self._grid_res, self._grid_res, self._grid_res)) * -1 * self._truncated_value
        x, y, z = np.where(sdf_final)
        points_model_grid = (x, y, z)
        points_model_grid = np.concatenate((np.array(points_model_grid).transpose(1, 0), np.ones((len(x), 1))), axis=1)
        points_sdf_grid = points_model_grid @ np.matrix.transpose(TshapeNetGrid2sdfGrid)
        valid_points = []
        valid_points_idx = []

        x_list = points_sdf_grid[:, 0]
        y_list = points_sdf_grid[:, 1]
        z_list = points_sdf_grid[:, 2]
        idx_x = np.where((x_list >= 0) & (x_list < sample_sdf.dimx - 1))
        idx_y = np.where((y_list >= 0) & (y_list < sample_sdf.dimy - 1))
        idx_z = np.where((z_list >= 0) & (z_list < sample_sdf.dimz - 1))
        valid_points_idx = np.intersect1d(np.intersect1d(idx_x[0], idx_y[0]), idx_z[0])
        valid_points = points_sdf_grid[valid_points_idx][:, :3]

        sdf_model = interpolate.interpn(sdf_grid_coords, sdf_vox, valid_points)
        x_sdf = np.array(points_model_grid[valid_points_idx][:, 0], dtype=np.int)
        y_sdf = np.array(points_model_grid[valid_points_idx][:, 1], dtype=np.int)
        z_sdf = np.array(points_model_grid[valid_points_idx][:, 2], dtype=np.int)
        sdf_final[x_sdf, y_sdf, z_sdf] = sdf_model

        # get mask
        # instance_mask = self.get_instance_mask(obj_file, voxel_size, voxel_origin, Mbbox2model, Tscales)
        
        return sdf_final, mask_file

    def get_instance_mask(self, obj_file, voxel_size, voxel_origin, Mbbox2model, Tscales):
        instance_mesh = trimesh.load(obj_file, process=False)
        # convert bbox mesh to points coordinate
        bbox_mesh_model = self._bbox_mesh.copy()
        bbox_mesh_model.apply_transform(Mbbox2model)
        bbox_mesh_model.apply_transform(Tscales)
        instance_voxel = instance_mesh.voxelized(voxel_size)
        bbox_voxel = bbox_mesh_model.voxelized(voxel_size)
        # get prepad for instance and bbox voxels corresponding to 32 cube
        instance_det = instance_voxel.origin - voxel_origin
        instance_start = np.floor(instance_det / voxel_size).astype(np.int64)
        bbox_det = bbox_voxel.origin - voxel_origin
        bbox_start = np.floor(bbox_det / voxel_size).astype(np.int64)
        bbox_end = bbox_start + bbox_voxel.shape
        bbox_start -= self._voxel_extend
        bbox_end += self._voxel_extend
        bbox_start[np.where(bbox_start < 0)] = 0
        bbox_end[np.where(bbox_end > int(self._grid_res-1))] = int(self._grid_res-1)

        x, y, z = np.where(instance_voxel.matrix)
        instance_mask = np.zeros((self._grid_res, self._grid_res, self._grid_res))

        x = np.array(x) + instance_start[0]
        y = np.array(y) + instance_start[1]
        z = np.array(z) + instance_start[2]
        idx_x = np.where((x >= bbox_start[0]) & (x <= bbox_end[0]))
        idx_y = np.where((y >= bbox_start[1]) & (y <= bbox_end[1]))
        idx_z = np.where((z >= bbox_start[2]) & (z <= bbox_end[2]))
        valid_points_idx = np.intersect1d(np.intersect1d(idx_x[0], idx_y[0]), idx_z[0])
        x_new = x[valid_points_idx]
        y_new = y[valid_points_idx]
        z_new = z[valid_points_idx]
        instance_mask[x_new, y_new, z_new] = 1
        return instance_mask

    def load_sample_sdf(self, filename):
        assert os.path.isfile(filename), "file not found: %s" % filename
        if filename.endswith(".df"):
            f_or_c = "C"
        else:
            f_or_c = "F"

        fin = open(filename, 'rb')
        
        s = SampleSDF()
        s.filename = filename
        s.dimx = struct.unpack('Q', fin.read(8))[0]
        s.dimy = struct.unpack('Q', fin.read(8))[0]
        s.dimz = struct.unpack('Q', fin.read(8))[0]
        s.res = struct.unpack('f', fin.read(4))[0]
        n_elems = s.dimx * s.dimy * s.dimz

        s.world2grid = struct.unpack('f'*16, fin.read(16*4))
        
        s.num_locations = struct.unpack('Q', fin.read(8))[0]
        try:
            location_bytes = fin.read(s.num_locations*3*4)
            s.locations = struct.unpack('I'*3*s.num_locations, location_bytes)
            
            sdfs_bytes = fin.read(s.num_locations*4)
            s.sdfs = struct.unpack('f'*s.num_locations, sdfs_bytes)
        except struct.error:
            print("Cannot load", filename)
        
        fin.close()
        s.world2grid = np.asarray(s.world2grid, dtype=np.float32).reshape([4, 4], order=f_or_c)
        s.locations = np.asarray(s.locations, dtype=np.uint32).reshape([s.num_locations, 3], order="C")
        s.sdfs = np.array(s.sdfs, dtype=np.float32)

        return s

    def load_sample_knw(self, filename):
        assert os.path.isfile(filename), "file not found: %s" % filename
        if filename.endswith(".df"):
            f_or_c = "C"
        else:
            f_or_c = "F"

        fin = open(filename, 'rb')
                
        s = SampleKNW()
        s.filename = filename
        s.dimx = struct.unpack('Q', fin.read(8))[0]
        s.dimy = struct.unpack('Q', fin.read(8))[0]
        s.dimz = struct.unpack('Q', fin.read(8))[0]
        s.res = struct.unpack('f', fin.read(4))[0]

        s.world2grid = struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))
        s.world2grid = np.asarray(s.world2grid, dtype=np.float32).reshape([4, 4])

        try:
            s.known = struct.unpack('B' * s.dimz * s.dimy * s.dimx, fin.read(s.dimz * s.dimy * s.dimx))
            s.known = np.asarray(s.known, dtype=np.uint8).reshape([s.dimz, s.dimy, s.dimx])
        except struct.error:
            print("Cannot load", filename)

        fin.close()
        # convert knows from zyx to xyz
        s.known = np.transpose(s.known, (2,1,0)).copy()
        return s

def main():
    tsdf_extractor = TsdfExtractor()
    tsdf_extractor.extract_data_multi_process()
    # print (tsdf_extractor._bad_scene)
    # print (tsdf_extractor._bad_cases)

if __name__ == "__main__":
    main()
