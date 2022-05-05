import argparse
import os
import glob
import json
import trimesh
import time
import copy
import numpy as np

import util

class InstanceMeshesGeneration:
    """
    This class corps labeled instance meshes based on ScanNet annotation and Scan2cad annotation
    """
    def __init__(self, args):
        """
        Initializes InstanceMeshesGeneration object. Sets up related pathes
        and other parameters.

        :param args: arguments from the commandline
        :type args: argparse.Namespace
        """
        self.args = args

        # init path for related files
        self._scene_path = self.args.scene_path
        if os.path.exists(self._scene_path) is not True:
            util.print_error('Please enter valid scene path')
            return
        self._gt_info_path = self.args.gt_info_path
        if os.path.exists(self._gt_info_path) is not True:
            util.print_error('Please enter valid gt info path')
            return

        # not use Scan2CAD annotation for cropping object mesh, then init
        # a list for the objects that do not need to consider
        self._labels_without_annotation = ['wall', 'floor', 'ceiling', 'window', 'door']

        # create output folder
        self._output_path = self.args.output_path
        os.makedirs(self._output_path, exist_ok=True)

        # get the min number of voxels for considering mask and mesh are aligned with each other
        self._overlap_vox_min = self.args.overlap_vox_min
        # a varialbe that used for checking whether the size of transferred mesh is reasonable or not (m)
        self._max_size = 100
        self._voxel_padding = 2

        # bbox mesh
        bbox_file = "~/research/Scan2CAD/Routines/Script/bbox.ply"
        self._bbox_mesh = trimesh.load(bbox_file)

        # a dict to map instance_id to cropped instance mesh for each scene
        self._instance_meshes = {}
        # a dict to map instance_id to its corresponding binary mask file for each scene
        self._instance_masks = {}
        self._bad_samples = []

    def generate_data(self):
        """
        This function generates training data form ScanNet data
        """
        start_time = time.time()
        # iterate all the scenes
        _, scene_names, _ = next(os.walk(self._gt_info_path))
        for scene_name in scene_names:
            print("processing: " + scene_name)
            # extract instances meshes based on binary masks
            self.extract_instance_meshes(scene_name)
            # align the instances meshes with binary masks
            self.alignment(scene_name)
            # save corresponding instance meshes
            self.save_meshes(scene_name)
            # reset related parameters
            self._instance_meshes = {}
            self._instance_masks = {}
        print ("costing time:")
        print (time.time() - start_time)

    def read_aggregation(self, filename):
        """
        This function reads aggregation file

        :param filename: aggration file name
        :type filename: str
        :return object_id_to_segs: segments for each object_id
        :rtype object_id_to_segs: dict
        :return segments_file: the corresponding segmentation file
        :rtype segments_file: str
        """
        object_id_to_segs = {}
        with open(filename) as f:
            data = json.load(f)
            num_objects = len(data['segGroups'])
            segments_file = data['segmentsFile']
            for i in range(num_objects):
                label = data['segGroups'][i]['label'].replace(' ', '-')
                # save objects with labels that might be annotated by scan2CAD
                if label not in self._labels_without_annotation:
                    object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
                    segs = data['segGroups'][i]['segments']
                    object_id_to_segs[object_id] = segs
        return object_id_to_segs, segments_file

    def read_segmentation(self, filename):
        """
        This function reads segmentation file for the scene

        :param filename: segmentation file name
        :type filename: str
        :return seg_to_verts: vertices for each segment
        :rtype seg_to_verts: dict
        :return num_verts: the number of vertex for this mesh
        :rtype num_verts: int
        """
        seg_to_verts = {}
        with open(filename) as f:
            data = json.load(f)
            num_verts = len(data['segIndices'])
            for i in range(num_verts):
                seg_id = data['segIndices'][i]
                if seg_id in seg_to_verts:
                    seg_to_verts[seg_id].append(i)
                else:
                    seg_to_verts[seg_id] = [i]
        return seg_to_verts, num_verts

    def get_instance_annotation(self, scene_name):
        """
        This function reads 3D instance annotation

        :param scene_name: scene name
        :type scene_name: str
        :return instance_ids: object id for each vertex
        :rtype instance_ids: 1d np.array
        """
        # get related file names
        scan_path = os.path.join(self._scene_path, scene_name)
        agg_file = os.path.join(scan_path, scene_name + '.aggregation.json')
        # read aggregation file
        if os.path.isfile(agg_file) is False:
            util.print_error(scene_name + " doesn't have correct aggregation file.")
            return []
        object_id_to_segs, segments_file = self.read_aggregation(agg_file)
        # read segmentation file
        seg_file = segments_file.split("scannet.")[-1]
        seg_file = os.path.join(scan_path, seg_file)
        if os.path.isfile(seg_file) is False:
            util.print_error(scene_name + " doesn't have correct segmentation file.")
            return []
        seg_to_verts, num_verts = self.read_segmentation(seg_file)
        # get instance ids for each vertex
        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                instance_ids[verts] = object_id
        return instance_ids

    def extract_instance_meshes(self, scene_name):
        """
        This function extracts meshes for all the instances

        :param scene_name: scene name
        :type scene_name: str
        """
        # get instance annotation
        instance_ids = self.get_instance_annotation(scene_name)
        if len(instance_ids) == 0:
            print ("No annotated instances can be found in " + scene_name)
            return
        # load scene mesh
        scene_mesh = None
        mesh_file = os.path.join(os.path.join(self._scene_path, scene_name), scene_name + '_vh_clean_2.ply')
        # No color info for loading by trimesh
        scene_mesh = trimesh.load(mesh_file, process=False)
        if len(scene_mesh.vertices) != len(instance_ids):
            util.print_error('#predicted vertex = ' + str(len(instance_ids)) + 'vs #mesh vertices = ' + str(len(scene_mesh.vertices)))

        # get instance meshes based on object id
        # meshes_vex_idx stores each mesh's vertices indexes, key is the instance_id and value is the list of vertices indexes
        # meshes_face_idx stores each mesh's faces indexes, key is the instance_id and value is the set of faces indexes
        meshes_vex_idx = {}
        meshes_face_idx = {}
        for idx, instance_id in enumerate(instance_ids):
            if instance_id != 0:
                if instance_id in meshes_vex_idx:
                    meshes_vex_idx[instance_id].append(idx)
                    for face_idx in scene_mesh.vertex_faces[idx]:
                        if face_idx != -1:
                            meshes_face_idx[instance_id].add(face_idx)
                else:
                    meshes_vex_idx[instance_id] = [idx]
                    meshes_face_idx[instance_id] = set()
                    for face_idx in scene_mesh.vertex_faces[idx]:
                        if face_idx != -1:
                            meshes_face_idx[instance_id].add(face_idx)

        # generate meshes based on corresponding vertices and faces
        for instance_id in meshes_vex_idx.keys():
            # generate map from vertex_old_index to vertex_new_index for faces genetration
            vertex_idx_map = {}
            # vertices list
            vertex_list = []
            for new_idx, old_idx in enumerate(meshes_vex_idx[instance_id]):
                vertex_idx_map[old_idx] = new_idx
                vertex_list.append(scene_mesh.vertices[old_idx])

            # get faces for this mesh
            face_list = []
            for face_idx in meshes_face_idx[instance_id]:
                # check whether all the vertices indexes are still kept
                face = scene_mesh.faces[face_idx]
                if face[0] in vertex_idx_map.keys() and face[1] in vertex_idx_map.keys() and face[2] in vertex_idx_map.keys():
                    face_list.append([vertex_idx_map[face[0]], vertex_idx_map[face[1]], vertex_idx_map[face[2]]])

            # generate mesh based on vertices and faces
            mesh = trimesh.Trimesh(vertices=vertex_list,faces=face_list, process=False)
            self._instance_meshes[instance_id] = mesh

    def alignment(self, scene_name):
        """
        This function aligns instances meshes to its corresponding binary masks for each scene

        :param scene_name: scene name
        :type scene_name: str
        """
        binary_masks_path = os.path.join(self._gt_info_path, scene_name)
        for mask_file in glob.iglob(os.path.join(binary_masks_path, "*mask.npz")):
            aligned_instance_id = self.align_mask(mask_file)
            if aligned_instance_id == -1:
                print ("Cannot find related instance mesh in scannet scene.")
                continue
            if aligned_instance_id in self._instance_masks.keys():
                self._instance_masks[aligned_instance_id].append(mask_file)
            else:
                self._instance_masks[aligned_instance_id] = [mask_file]

    def align_mask(self, mask_file):
        """
        This function aligns the binary mask with its corresponding instance in
        the scene based on annotation data form scan2CAD

        :param mask_file: gt info file
        :type mask_file: str
        :return aligned_instance_id: the instance id aligned with this mask
        :rtype aligned_instance_id: str
        """
        # get related data for mask and bbox
        with np.load(mask_file) as data:
            Mmodel2scannet = data['Mmodel2scannet']
            mask = data['mask']
            bbox_origin = data['voxel_origin']
            voxel_size = data['voxel_size']
            Mbbox2model = data['Mbbox2model']
            T_scales = data['T_scales']

        bbox_mesh_model = self._bbox_mesh.copy()
        bbox_mesh_model.apply_transform(Mbbox2model)
        bbox_mesh_model.apply_transform(T_scales)
        bbox_min = bbox_mesh_model.bounds[0]
        bbox_max = bbox_mesh_model.bounds[1]
        bbox_shape = np.ceil((bbox_max - bbox_min) / voxel_size).astype(np.int64)
        # init aligned instance id and the percentage about how many vertices are there in the bbox
        aligned_instance_id = -1
        overlap_vox_max = self._overlap_vox_min
        for instance_id, mesh in self._instance_meshes.items():
            # transfer instance mesh based on the transformation matrix
            mesh_trans = copy.deepcopy(mesh).apply_transform(np.linalg.inv(Mmodel2scannet))
            mesh_bounds = mesh_trans.bounds
            # check the size of the transferred mesh, make sure all the transfers are make sense 
            if max(mesh_bounds[1,:] - mesh_bounds[0,:]) > self._max_size:
                continue
            # check whether transferred mesh overlaps with the mask bbox
            if ((bbox_min[0] > mesh_bounds[1][0] or bbox_max[0] < mesh_bounds[0][0]) or
                (bbox_min[1] > mesh_bounds[1][1] or bbox_max[1] < mesh_bounds[0][1]) or
                (bbox_min[2] > mesh_bounds[1][2] or bbox_max[2] < mesh_bounds[0][2])):
                continue
            # mesh_trans.export("")
            # convert mesh_trans to voxel with voxel_size
            voxel = mesh_trans.voxelized(voxel_size)
            voxel_filled = voxel.copy().fill(method="orthographic")
            voxel_origin = voxel_filled.origin
            voxel_matrix = voxel_filled.encoding.dense
            voxel_max = voxel_origin + voxel.shape * voxel_size
            # get bbox voxel
            bbox_det = bbox_min - bbox_origin
            bbox_start = np.round(bbox_det / voxel_size).astype(np.int64)
            bbox_end = bbox_start + bbox_shape
            bbox_start[np.where(bbox_start < 0)] = 0
            bbox_end[np.where(bbox_end > 31)] = 31
            # combine mask and mesh_mask to union_matrix for getting the overlap area
            union_min = np.minimum(bbox_min, voxel_origin)
            union_max = np.maximum(bbox_max, voxel_max)
            union_matrix = np.zeros(np.ceil(1 + (union_max - union_min) / voxel_size).astype(np.int64)) # add 1 padding for easier overlapping
            # add mask to union_matrix
            mask_start = np.round((bbox_min - union_min) / voxel_size).astype(np.int64)
            mask_end = mask_start + bbox_shape
            for i in range(3):
                if mask_end[i] > union_matrix.shape[i]:
                    mask_end[i] = union_matrix.shape[i]
            try:
                union_matrix[mask_start[0] : mask_end[0], mask_start[1] : mask_end[1], 
                            mask_start[2] : mask_end[2]] += mask[bbox_start[0] : bbox_end[0],
                                                                 bbox_start[1] : bbox_end[1],
                                                                 bbox_start[2] : bbox_end[2]]
            except:
                self._bad_samples.append(mask_file)
            # add mesh to union_matrix
            mesh_idx = np.round((voxel_origin - union_min) / voxel_size).astype(np.int64)
            mesh_mask = np.zeros(voxel_matrix.shape)
            mesh_mask[np.where(voxel_matrix==True)] = 1
            union_matrix[mesh_idx[0] : mesh_idx[0] + mesh_mask.shape[0], mesh_idx[1] : mesh_idx[1] + mesh_mask.shape[1], mesh_idx[2] : mesh_idx[2] + mesh_mask.shape[2]] += mesh_mask
            # get how many voxels overlap with the mask
            overlap_vox = np.sum(union_matrix == 2)
            # print (instance_id, overlap_vox)
            if overlap_vox > overlap_vox_max:
                overlap_vox_max = overlap_vox
                aligned_instance_id = instance_id

        return aligned_instance_id

    def save_meshes(self, scene_name):
        """
        This function saves instance meshes based on aligned binary masks

        :param scene_name: scene name
        :type scene_name: str
        """
        output_path = os.path.join(self._output_path, scene_name)
        for instance_id, mask_files in self._instance_masks.items():
            if len(mask_files) > 1:
                print ("Cannot find 1-to-1 mapping for instance: {}".format(instance_id))
                continue
            mask_file = mask_files[0]
            # print (mask_file)
            with np.load(mask_file) as data:
                Mmodel2scannet = data['Mmodel2scannet']
                mesh = self._instance_meshes[instance_id]
                mesh_trans = copy.deepcopy(mesh).apply_transform(np.linalg.inv(Mmodel2scannet))
                output_name = os.path.join(output_path, str(instance_id - 1) + '_' +
                                           mask_file.split('/')[-1].split('.')[0] + "_scaled.obj") # instance ids should be 0-indexed
                mesh_trans.export(output_name)

def parse_arguments():
    """
    Generates a command line parser that supports all arguments used by the
    tool.

    :return: Parser with all arguments.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    # required parameters
    parser.add_argument("--scene_path",
                        help="path for scene data",
                        required=True,
                        type=str)
    parser.add_argument("--gt_info_path",
                        help="path for gt info",
                        required=True,
                        type=str)
    # optional parameters
    parser.add_argument("--output_path",
                        help="path to save the training files",
                        default="data_samples/scannet",
                        type=str)
    parser.add_argument("--overlap_vox_min",
                        help="the min number of voxels for considering mask and mesh"
                             " are aligned with each other",
                        default=10,
                        type=int)

    return parser


def main():
    """
    generates instance meshes for ScanNet dataset
    """
    arg_parser = parse_arguments()
    instance_meshes_generator = InstanceMeshesGeneration(arg_parser.parse_args())
    # generate and save cropped instance meshes for ScanNet dataset
    instance_meshes_generator.generate_data()


if __name__ == "__main__":
    main()
