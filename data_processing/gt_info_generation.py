import argparse
import os
import csv
import glob
import json
import time
import trimesh
import quaternion
import copy
import numpy as np
import math
import IPython

import util

class GTGeneration:
    """
    This class generates GT info for each objects based on Scan2CAD annotation and shapeNet models

    For each scene
    1. Map shapeNet models to instances in the scene
    2. Generate gt info for these aligned shapeNet models 
    """
    def __init__(self, args):
        """
        Initializes GTGeneration object. Sets up related pathes and other parameters.

        :param args: arguments from the commandline
        :type args: argparse.Namespace
        """
        self.args = args

        # init path for related files
        self._annotation_file = self.args.annotation_file
        if os.path.exists(self._annotation_file) is not True:
            util.print_error('Please enter valid annotation file from scan2cad')
            return
        self._shapeNet_path = self.args.shapeNet_path
        if os.path.exists(self._shapeNet_path) is not True:
            util.print_error('Please enter valid shapeNet path')
            return
        self._map_file = self.args.map_file
        if os.path.exists(self._map_file) is not True:
            util.print_error('Please enter valid map file for shapeNet and scanNet id')
            return
        self._bbox_mesh_file = self.args.bbox_mesh_file
        if os.path.exists(self._bbox_mesh_file) is not True:
            util.print_error('Please enter valid bbox mesh file for scan2CAD')
            return
        # create output folder
        self._output_path = self.args.output_path
        os.makedirs(self._output_path, exist_ok=True)

        # parameters for generating binary masks
        self._cube_dimension = self.args.cube_dimension
        self._bbox_padding = 2
        self._cube_dimension -= 2 * self._bbox_padding
        if (self._cube_dimension < 0):
            util.print_error('Please enter cub_dimension or padding')
            return

        # read annotation data
        with open(self._annotation_file) as annotation_file:
            self._annotation = json.load(annotation_file)

        # map scanNet label to shapeNet cadid
        self._id_map = util.map_label_to_cadid(self._map_file)

        # a dict for mapping instance_id to its mesh
        self._instance_meshes = {}
        # a dict for mapping instance_id to its mesh bbox bounds
        self._instance_bbox_bounds = {}
        # get original bbox mesh for scan2CAD
        self._bbox_mesh = trimesh.load(self._bbox_mesh_file, process=False)
        # parameters for threshold of vertices in a bbox
        self._vert_percentage = self.args.vert_percentage

    def generate_data(self):
        """
        This function generates binary masks from aligned shapeNet models
        """
        start_time = time.time()
        # iterate all the scenes
        for data in self._annotation:
            scene_name = data['id_scan']
            print ("procsessing: " + scene_name)
            # align shapeNet models with corresponding instance meshes from scanNet 
            self.generate_scene_masks(scene_name, data)
            # reset related parameters
            self._instance_meshes = {}
            self._instance_bbox_bounds = {}

        print ("costing time:")
        print (time.time() - start_time)

    def generate_scene_masks(self, scene_name, data):
        """
        This function generates binary masks for instances in each scene

        :param scene_name: scene name
        :type scene_name: str
        :param data: annotation data for this scene
        :type data: dict
        """
        # set up saving directory
        output_path = os.path.join(self._output_path, scene_name)
        os.makedirs(output_path, exist_ok=True)
        # get transformation matrix between scanNet frame to shapeNet frame
        Mscan2shape = self.make_M_from_tqs(data["trs"]["translation"], data["trs"]["rotation"], data["trs"]["scale"])

        # iterate models in annotation data
        for idx, model in enumerate(data['aligned_models']):
            # get shapeNet model
            model_file = os.path.join(
                            os.path.join(
                                os.path.join(
                                    self._shapeNet_path, model['catid_cad']),
                                    model['id_cad']), 
                                    "models/model_normalized.obj")
            # load shapeNet model mesh
            model_mesh = trimesh.load(model_file, procsessing=False)
            if type(model_mesh) == trimesh.scene.scene.Scene:
                model_mesh = trimesh.util.concatenate(model_mesh.dump())

            # transfer and scale this model in shapeNet frame
            # transformation matrix to convert model to corresponding scannet object size
            Mmodel2real = self.make_M_from_tqs(model["trs"]["translation"], model["trs"]["rotation"], model["trs"]["scale"])
            # transforamtion matrix between shapeNet model to scanNet frame
            Mmodel2scannet = np.matmul(np.linalg.inv(Mscan2shape), Mmodel2real)

            # get transformation matrix for bbox to model
            Mbbox2scannet = self.get_M_bbox(Mmodel2scannet, model)
            Mbbox2model= np.linalg.inv(Mmodel2scannet).dot(Mbbox2scannet)
            # get transformed annotation bbox
            bbox_mesh = copy.deepcopy(self._bbox_mesh)
            model_bbox_model = bbox_mesh.apply_transform(Mbbox2model)
            
            # get Mmodel2scannet without scalling
            Tscales = np.identity(4)
            for i in range(3):
                vec = Mmodel2scannet[:3, i]
                s = sum([e * e for e in vec])
                scale = math.sqrt(s)
                Mmodel2scannet[:3, i] /= scale
                Tscales[i, i] = scale
            model_mesh.apply_transform(Tscales)
            model_bbox_model.apply_transform(Tscales)
            
            # save gt info
            self.save_gt_info(idx, output_path, copy.deepcopy(model_mesh),
                              copy.deepcopy(model_bbox_model), Mmodel2scannet,
                              Mbbox2model, model['catid_cad'], model['id_cad'],
                              Tscales)

    def save_gt_info(self, idx, output_path, model, bbox,
                     Mmodel2scannet, Mbbox2model, cat_id, type_id, T_scales):
        """
        This function saves binary mask for this instance

        :param idx: idx for this model
        :type idx: str
        :param output_path: path to save binary masks
        :type output_path: str
        :param model: model mesh
        :type model: trimesh.Trimesh
        :param bbox: transferred bbox to shapenet frame
        :type bbox: trimesh.Trimesh
        :param Mmodel2scannet: transfermation matrix from shapenet frame to scannet frame
        :type Mmodel2scannet: np.array
        :param Mbbox2model: transfermation matrix from bbox frame to shapenet frame
        :type Mbbox2model: np.array
        :param cat_id: category id for this model
        :type cat_id: str
        :param type_id: type id for this model
        :type type_id: str
        """ 
        # save related info and binary mask for this model
        gt_info_name = os.path.join(output_path, cat_id + "_" + type_id + "_" + str(idx) + "_mask.npz")
        if os.path.exists(gt_info_name):
            return
        # generate binary mask and related info for this instance
        voxel_origin, voxel_size = self.generate_gt_info_with_bbox(
                                            copy.deepcopy(model),
                                            bbox.bounds,
                                            bbox.centroid)
        np.savez(gt_info_name,
                 T_scales=T_scales,
                 Mmodel2scannet=Mmodel2scannet,
                 Mbbox2model=Mbbox2model,
                 voxel_size=voxel_size,
                 voxel_origin=voxel_origin)#,
                #  mask=binary_mask)
        print (gt_info_name)

    def generate_gt_info_with_bbox(self, mesh, bounds, centroid):
        """
        This function voxelize a mesh based on bounds and centroid of a bbox

        :param mesh: the mesh that needs to be transferred to voxel
        :type mesh: trimesh.Trimesh
        :param bounds: bounds of bbox
        :type bounds: list
        :param centroid: centroid of bbox
        :type centroid: list
        :return mask: binary mask for this mesh based on bounds of the bbox
        :rtype mask: a 3D np.array
        :return local_origin: origin of voxels before transfrom
        :rtype local_origin: list
        :return voxel_size: voxel_size for generating binary mask
        :rtype voxel_size: float
        """
        # translate mesh based on the center of bbox
        mesh.apply_translation(-centroid)
        # get voxel size for generating cube
        voxel_size = max(bounds[:][1] - bounds[:][0]) / self._cube_dimension
        # origin of local voxels
        local_origin = centroid - (self._cube_dimension + 2 * self._bbox_padding)/ 2 * voxel_size

        '''
        # save mask
        voxel_size = max(mesh.bounds[:][1] - mesh.bounds[:][0]) / self._cube_dimension
        voxel_size = np.round(voxel_size, 2) if voxel_size < np.round(voxel_size, 2) else np.round(voxel_size, 2) + 0.01
        print (voxel_size)
        voxelize translated mesh
        voxel = mesh.voxelized(voxel_size)
        voxel_filled = voxel.copy().fill(method="orthographic")
        origin = voxel_filled.transform[:3, 3]
        matrix = voxel_filled.encoding.dense

        Find voxel index for point
        center_idx = np.round(-origin / voxel_size).astype(np.int64)
        get pad number for mesh voxels
        prepad = np.maximum(int(self._cube_dimension / 2) - center_idx, 0) + [self._bbox_padding, self._bbox_padding, self._bbox_padding]
        postpad = np.maximum(center_idx + int(self._cube_dimension / 2) - matrix.shape, 0) + [self._bbox_padding, self._bbox_padding, self._bbox_padding]
        # pad matrix if necessary
        matrix_padded = np.pad(matrix, np.stack((prepad, postpad), axis=-1), mode='constant')
        mask = np.zeros(matrix_padded.shape)
        mask[np.where(matrix_padded==True)] = 1
        '''

        return local_origin, voxel_size

    def make_M_from_tqs(self, t, q, s):
        """
        This function generates transformation matrix based on trans, rotation and scale

        :param t: trans
        :type t: np.array
        :param q: quaternion
        :type t: np.array
        :param s: scale
        :type t: np.array
        :return M: transformation matrix
        :rtype M: np.arrays
        """
        q = np.quaternion(q[0], q[1], q[2], q[3])
        T = np.eye(4)
        T[0:3, 3] = t
        R = np.eye(4)
        R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
        S = np.eye(4)
        S[0:3, 0:3] = np.diag(s)

        M = T.dot(R).dot(S)
        return M 

    def get_M_bbox(seld, trans_matrix, model):
        """
        This function generates the transformed and scaled bbox based on the scan2CAD annotations

        :param trans_matrix: transformation + scale matrix from shapeNet model to scanNet scene
        :type trans_matrix: np.array
        :param model: annotation for this model
        :paran model: dict
        :return Mbbox2scannet: tranformation and scale matrix for bbox from scan2CAD to scannet scene
        :rtype Mbbox2scannet: np.array
        """
        bbox = np.asarray(model["bbox"], dtype=np.float64)
        center = np.asarray(model["center"], dtype=np.float64)
        center_mat = np.eye(4)
        center_mat[0:3, 3] = center
        bbox_mat = np.eye(4)
        bbox_mat[0:3, 0:3] = np.diag(bbox)
        Mbbox2scannet = trans_matrix.dot(center_mat).dot(bbox_mat)

        return Mbbox2scannet


def parse_arguments():
    """
    Generates a command line parser that supports all arguments used by the
    tool.

    :return: Parser with all arguments.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    # required parameters
    parser.add_argument("--annotation_file",
                        help="annotation file from scan2CAD",
                        required=True,
                        type=str)
    parser.add_argument("--shapeNet_path",
                        help="path for shapeNet model data",
                        required=True,
                        type=str)
    parser.add_argument("--map_file",
                        help="csv file for mapping scanNet label to shapeNet catid",
                        required=True,
                        type=str)
    parser.add_argument("--bbox_mesh_file",
                        help="bbox mesh file for scan2CAD annotation",
                        required=True,
                        type=str)
    # optional parameters
    parser.add_argument("--output_path",
                        help="path to save the training files",
                        default="/mnt/login_cluster/gimli/yrao/output_new_coords/",
                        type=str)
    parser.add_argument("--cube_dimension",
                        help="cube dimension for generated voxel grids from mesh",
                        default=32,
                        type=int)
    parser.add_argument("--vert_percentage",
                        help="threshold for considering whehter a mesh is overlaps with a bbox,"
                             " it used for the percentage of how many vertices of a mesh in a bbox",
                        default=0.5,
                        type=float)

    return parser


def main():
    """
    generates binary masks for training
    """
    arg_parser = parse_arguments()
    gt_generator = GTGeneration(arg_parser.parse_args())
    # generate and save binary masks based on aligned shapeNet models
    gt_generator.generate_data()


if __name__ == "__main__":
    main()