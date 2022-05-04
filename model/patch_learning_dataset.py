import os
import glob
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh

class ShapenetDataset(Dataset):
    """
    This class reads all the training dataset for Shapenet
    """
    def __init__(self, file_name, data_path, res=32, truncation=2.5):
        """
        Initializes ShapenetDataset object. Sets up related pathes
        and other parameters.

        :param file: file contains trainig sample names
        :type file: str
        :param data_path: path for the data
        :type data_path: str
        :param res: input resolution
        :type res: int
        :param truncation: value for truncation
        :type truncation: float
        """
        # path parameters
        self._file = file_name
        self._data_path = data_path
        # read data
        self._data_pairs = []
        self._mask_names = []
        self._bbox = []
        self._truncation = truncation
        self._res = res
        self._read_data()

    def get_bbox(self, gt):
        x1 = 0
        x2 = self._res -1
        y1 = 0
        y2 = self._res -1
        z1 = 0
        z2 = self._res -1
        for i in range(self._res):
            if len(np.where(gt[i, :, :]<= 0)[0]) > 0:
                x1 = i
                break
        for i in range(self._res):
            if len(np.where(gt[:, i, :]<= 0)[0]) > 0:
                y1 = i
                break
        for i in range(self._res):
            if len(np.where(gt[:, :, i]<= 0)[0]) > 0:
                z1 = i
                break
        for i in range(self._res - 1, 0, -1):
            if len(np.where(gt[i, :, :]<= 0)[0]) > 0:
                x2 = i
                break
        for i in range(self._res - 1, 0, -1):
            if len(np.where(gt[:, i, :]<= 0)[0]) > 0:
                y2 = i
                break
        for i in range(self._res - 1, 0, -1):
            if len(np.where(gt[:, :, i]<= 0)[0]) > 0:
                z2 = i
                break
        bbox = [[x1, x2], [y1, y2], [z1, z2]]
        return bbox

    def _read_data(self):
        """
        This function reads data from data path
        """
        with open(self._file, 'r') as data:
            print (self._file)
            gt_files = data.readlines()
            for gt_file in gt_files:
                model_path = os.path.join(self._data_path, gt_file.split('\n')[0])
                gt_file = os.path.join(model_path, "gt.npz")
                with np.load(gt_file, 'rb') as data:
                    gt = data["tsdf"] * self._res # convert to voxel unit
                    gt[np.where(gt>self._truncation)] = self._truncation
                    gt[np.where(gt<-1*self._truncation)] = -1*self._truncation
                    bbox = self.get_bbox(gt)
                for input_file in glob.glob(os.path.join(model_path, "input*.npz")):
                    with np.load(input_file, 'rb') as data:
                        inputs = data["tsdf"] * self._res
                        inputs[np.where(inputs>self._truncation)] = self._truncation
                        inputs[np.where(inputs<-1 * self._truncation)] = -1 * self._truncation
                        self._data_pairs.append([inputs, gt])
                        self._mask_names.append(input_file)
                        self._bbox.append(np.array(bbox))
        print (len(self._data_pairs))

    def __len__(self):
        """
        This function returns the number of traing samples
        """
        return len(self._data_pairs)

    def __getitem__(self, idx):
        """
        This function returns idx-th sdf and labels in the dataset
        """
        name = self._mask_names[idx]
        input_sdf = copy.deepcopy(self._data_pairs[idx][0])
        gt_sdf = copy.deepcopy(self._data_pairs[idx][1])
        bbox = torch.from_numpy(self._bbox[idx]).unsqueeze(0).float()
        # get final data
        input_sdf = torch.from_numpy(input_sdf).unsqueeze(0).float()
        gt_sdf = torch.from_numpy(gt_sdf).unsqueeze(0).float()
        return [input_sdf, gt_sdf, bbox, name]

class ScannetDataset(Dataset):
    """
    This class reads all the training dataset for Scannet
    """
    def __init__(self, file_name, data_path, truncation=3, res=32, use_bbox=True):
        """
        Initializes ScannetDataset object. Sets up related pathes
        and other parameters.

        :param file: file contains scene names
        :type  file: str
        :param data_path: path for the data
        :type data_path: str
        :param truncation: value for truncation
        :typr truncation: float
        :param res: input resolution
        :type res: int
        :param use_bbox: whether using bbox to crop inputs or not
        :type use_bbox: bool
        """
        # path parameters
        self._file = file_name
        self._data_path = data_path
        self._truncation = truncation
        self._use_bbox = use_bbox
        self._res = res
        self._bbox_mesh = trimesh.load('~/research/Scan2CAD/Routines/Script/bbox.ply', process=False) # scan2CAD bbox mesh
        # read data
        self._input_sdf = []
        self._gt_sdf = []
        self._mask_names = []
        self._bbox = []
        self._read_data()

    def get_bbox_anno(self, mask_file):
        bbox_mesh = copy.deepcopy(self._bbox_mesh)
        with np.load(mask_file) as data:
            voxel_size = data["voxel_size"]
            voxel_origin=data['voxel_origin']
            Mbbox2model=data["Mbbox2model"]
            Tscales=data["T_scales"]
        model_bbox_model = bbox_mesh.apply_transform(Mbbox2model)
        model_bbox_model.apply_transform(Tscales)
        bbox_ori = model_bbox_model.bounds[0] - voxel_origin
        bbox_ori /= voxel_size
        bbox_ori = np.array(bbox_ori, dtype=int)
        bbox_end = model_bbox_model.bounds[1] - voxel_origin
        bbox_end /= voxel_size
        bbox_end = np.array(bbox_end, dtype=int)
        bbox_end += 1
        bbox_ori += 1
        bbox_ori[bbox_ori < 0] = 0
        bbox_end[bbox_end > self._res - 1] = self._res - 1

        return [[bbox_ori[0], bbox_end[0]], [bbox_ori[1], bbox_end[1]], [bbox_ori[2], bbox_end[2]]], voxel_size

    def _read_data(self):
        """
        This function reads data from data path
        """
        with open(self._file, 'r') as data:
            print (self._file)
            mask_files = data.readlines()
            voxel_size = 0.0
            for mask_file in mask_files:
                mask_file = os.path.join(self._data_path, mask_file.split('\n')[0])
                sdf_file = mask_file[:-4] + "_sdf.npz"
                gt_sdf_file = mask_file[:-8] + "scaled_sdf_gt.npy"
                if (os.path.exists(mask_file) is False or 
                    os.path.exists(sdf_file) is False or
                    os.path.exists(gt_sdf_file) is False):
                    continue
                # get bbox
                bbox, voxel_size = self.get_bbox_anno(mask_file)
                self._bbox.append(bbox)
                # get input data
                with np.load(sdf_file, 'rb') as sdf:
                    input_sdf = sdf["instance_sdf"]
                    input_sdf /= voxel_size # convert to voxel unit
                    input_sdf[np.where(input_sdf>self._truncation)] = self._truncation
                    input_sdf[np.where(input_sdf<-1*self._truncation)] = -1*self._truncation
                    if self._use_bbox:
                        input_sdf_new = np.ones(input_sdf.shape) * self._truncation
                        input_sdf_new[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = input_sdf[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
                        input_sdf = copy.deepcopy(input_sdf_new)
                    self._input_sdf.append(input_sdf)
                # get gt data
                with open(gt_sdf_file, 'rb') as data:
                    gt_sdf = np.load(data)
                    gt_sdf /= voxel_size
                    gt_sdf[np.where(gt_sdf>self._truncation)] = self._truncation
                    gt_sdf[np.where(gt_sdf< -1 * self._truncation)] = -1 * self._truncation
                    self._gt_sdf.append(gt_sdf)
                self._mask_names.append(mask_file)
        print (len(self._mask_names))

    def __len__(self):
        """
        This function returns the number of traing samples
        """
        return len(self._mask_names)

    def __getitem__(self, idx):
        """
        This function returns idx-th sdf and labels in the dataset
        """
        input_sdf = copy.deepcopy(self._input_sdf[idx])
        gt_sdf = copy.deepcopy(self._gt_sdf[idx])
        input_sdf = torch.from_numpy(input_sdf).unsqueeze(0).float()
        gt_sdf = torch.from_numpy(gt_sdf).unsqueeze(0)
        bbox = self._bbox[idx]
        return [input_sdf, gt_sdf, bbox, self._mask_names[idx]]

