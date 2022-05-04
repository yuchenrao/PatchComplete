import numpy as np
import os
import glob
import time
import argparse
import copy
import mcubes, trimesh
import torch
from chamfer_distance import ChamferDistance


class Evaluation:
    """
    This class caluate Chamfer Distance and IOU between GTs and predictions

    Steps:
    1. Read GT and prediction
    2. Convert voxels to meshes
    3. Sample points on the meshes and then calcualte chamfer distance
    4. Calculate IOU and F1
    """
    def __init__(self, args):
        """
        Initializes Evaluation object. Sets up related pathes and other parameters.

        :param args: arguments from the commandline
        :type args: argparse.Namespace
        """
        # init path for related files
        self._root = args.root
        if os.path.exists(self._root) is not True:
            print ('Please enter valid data root path')
            return
        self._pred_path = args.pred_path
        if os.path.exists(self._pred_path) is not True:
            print('Please enter valid pred path')
            return
        self._test_file = args.test_file
        if os.path.exists(self._test_file) is not True:
            print('Please enter valid test file path')
            return
        
        # set paths based on dataset
        self._data_path = os.path.join(self._root, args.dataset)
        self._pred_path = os.path.join(self._pred_path, args.dataset)
        # get output file
        self._points_n = self.args.points_n
        self._chamfer_dist = ChamferDistance()

    def calculate_iou(self, gt, pred, threshold):
        bool_true_voxels = gt > threshold
        bool_pred_voxels = pred > threshold
        total_union = (bool_true_voxels | bool_pred_voxels).sum()
        total_intersection = (bool_true_voxels & bool_pred_voxels).sum()
        return (total_intersection / total_union)

    def calculate_f1(self, gt_mask, pred_mask):
        mask_t = copy.deepcopy(gt_mask)
        mask_t[np.where(pred_mask==1)] += 10
        miss = len(np.where(mask_t==1)[0]) / np.sum(gt_mask)
        redundant = len(np.where(mask_t==10)[0]) / np.sum(gt_mask)
        f1 = np.sum(np.logical_and(gt_mask, pred_mask)) / (np.sum(np.logical_and(gt_mask, pred_mask)) + 0.5 * np.sum(np.logical_xor(gt_mask, pred_mask)))  
        return miss, redundant, f1

    def calculate_cd(self, gt_mask, pred_mask):
        # get points for prediction
        if np.sum(pred_mask) == 0:
            pred_points = np.zeros((self._points_n, 3))
        else:
            pred_points, _ = self.get_surface_points(pred_mask, 0, self._points_n, 32)
        # get points for gt
        gt_points, _ = self.get_surface_points(gt_mask, 0, self._points_n, 32)
        # calcualte CD
        gt_points_torch = torch.from_numpy(gt_points).cuda().unsqueeze(0).float() 
        pred_points_torch = torch.from_numpy(pred_points).cuda().unsqueeze(0).float()
        dist1, dist2 = self._chamfer_dist(gt_points_torch, pred_points_torch)
        # loss = (torch.mean(dist1)) + (torch.mean(dist2))
        eps = 1e-10
        loss = torch.sqrt(dist1 + eps).mean(1) + torch.sqrt(dist2 + eps).mean(1)
        return loss.detach().cpu().numpy()

    def get_surface_points(self, V, threshold, voxel_res):
        # padding
        logits = np.pad(V, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        vertices, triangles = mcubes.marching_cubes(logits, threshold)
        # recale to [0,1]
        vertices -= 1
        step = 1/(voxel_res-1)
        vertices = np.multiply(vertices, step)
        mesh = trimesh.Trimesh(vertices, triangles)
        points = mesh.sample(self._points_n)

        return points, mesh

    def crop_32(self, pred, mask_file):
        bbox_file = mask_file[:-4]+'_bbox.npy'
        bbox = np.load(bbox_file)
        bbox_ori = bbox[:, 0]
        bbox_end = bbox[:, 1]
        pred_crop = np.zeros(pred.shape)
        pred_crop[bbox_ori[0]:bbox_end[0], bbox_ori[1]:bbox_end[1], bbox_ori[2]:bbox_end[2]] = pred[bbox_ori[0]:bbox_end[0], bbox_ori[1]:bbox_end[1], bbox_ori[2]:bbox_end[2]]
        return pred_crop

    def evaluate_shapenet(self):
        """
        This function gets final cd evaluation results
        """
        start_time = time.time()
        eval_res = {}
        # iterate all the files
        with open(self._test_file, 'r') as data:
            gt_files = data.readlines()
            for gt_file in gt_files:
                gt_file = gt_file.split('\n')[0]
                pred_model_path = os.path.join(self._pred_path, gt_file)
                gt_model_path = os.path.join(self._data_path, gt_file)
                gt_name = os.path.join(gt_model_path, "gt.npz") 
                with np.load(gt_name, 'rb') as data:
                    gt = data['tsdf']
                    gt_mask = np.zeros(gt.shape)
                    gt_mask[np.where(gt<=1e-10)] = 1
                for pred_name in glob.glob(os.path.join(pred_model_path, "*_pred.npz")): 
                    with np.load(pred_name, 'rb') as data:
                        pred = data['predicted_voxels']
                        pred_mask = np.zeros(pred.shape)
                        pred_mask[np.where(pred<=1e-10)] = 1
                    # evaluate IOU and f1
                    iou = self.calculate_iou(gt_mask, pred_mask, 0.5)
                    miss, redundant, f1 = self.calculate_f1(gt_mask, pred_mask)
                    cd = self.calculate_cd(gt_mask, pred_mask)
                    eval_res[pred_name] = [cd, iou, miss, redundant, f1, pred_name]
                    
        print ("costing time:")
        print (time.time() - start_time)
        return eval_res

    def evaluate_scannet(self):
        """
        This function gets final cd evaluation results
        """
        start_time = time.time()
        iou_res = {}
        # iterate all the files
        with open(self._test_file, 'r') as data:
            gt_files = data.readlines()
            for gt_file in gt_files:
                gt_file = gt_file.split('\n')[0]
                mask_file = os.path.join(self._root, gt_file)
                gt_sdf_file = gt_file[:-8] + "scaled_gt_sdf.npy"
                gt_pts_name = os.path.join(self._root, gt_sdf_file)
                if os.path.exists(os.path.join(self._root, gt_file)) is False:
                    continue
                with np.load(os.path.join(self._root, gt_file), 'rb') as data:
                    voxel_size = data["voxel_size"]
                with open(gt_pts_name, 'rb') as data:
                    gt = np.load(data)
                    gt /= voxel_size
                    gt_mask = np.zeros(gt.shape)
                    gt_mask[np.where(gt<=1e-10)] = 1
                pred_pts_name = os.path.join(self._pred_path, gt_file[:-4] + '_pred.npz') 
                if os.path.exists(pred_pts_name) is False:
                    continue
                with np.load(pred_pts_name, 'rb') as data:
                    pred = data['predicted_voxels'][0]
                    pred_mask = np.zeros(pred.shape)
                    pred_mask[np.where(pred<=1e-10)] = 1
                    pred_mask = self.crop_32(pred_mask, mask_file)
                iou = self.calculate_iou(gt_mask, pred_mask, 0.4)
                mask_t = copy.deepcopy(gt_mask)
                mask_t[np.where(pred_mask==1)] += 10
                miss = len(np.where(mask_t==1)[0]) / np.sum(gt_mask)
                redundant = len(np.where(mask_t==10)[0]) / np.sum(gt_mask)
                f1 = np.sum(np.logical_and(gt_mask, pred_mask)) / (np.sum(np.logical_and(gt_mask, pred_mask)) + 0.5 * np.sum(np.logical_xor(gt_mask, pred_mask)))
                iou_res[pred_pts_name] = [iou, miss, redundant, f1, gt_pts_name]
        
        print ("costing time:")
        print (time.time() - start_time)
        return iou_res
    
def parse_arguments():
    """
    Generates a command line parser that supports all arguments used by the
    tool.

    :return: Parser with all arguments.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    # required parameters
    parser.add_argument("--dataset",
                        help="dataeset to be trained",
                        default="shapenet",
                        required=True,
                        type=str)
    parser.add_argument("--root",
                        help="root for reading and saving data",
                        default='data_samples',
                        type=str)
    parser.add_argument("--pred_path",
                        help="path for reading mesh data",
                        default='output',
                        type=str)
    # optional parameters
    parser.add_argument("--test_file",
                        help="test samples names",
                        default="test.txt",
                        type=str)
    parser.add_argument("--points_n",
                        help="the number of sampling points for chamfer distance",
                        default=10240,
                        type=int)
    return parser

def print_results(ious):
    cats_iou = {}
    for name, iou in ious.items():
        cat = name.split('/')[-3]
        if cat in cats_iou:
            cats_iou[cat].append(iou)
        else:
            cats_iou[cat] = [iou]
    sum_cd = 0
    sum_iou = 0
    sum_miss = 0
    sum_red = 0
    sum_f1 = 0
    sum_len = 0
    cat_ious = []
    cat_cds = []
    cat_misses = []
    cat_reds = []
    cat_f1s = []
    for cat, data_l in cats_iou.items():
        print (cat)
        print (len(data_l))
        cat_cd = []
        cat_iou = []
        cat_miss = []
        cat_red = []
        cat_f1 = []
        names = []
        for cd, iou, miss, red, f1, name in data_l:
            sum_cd += cd
            sum_iou += iou
            sum_miss += miss
            sum_red += red
            sum_f1 += f1
            cat_cd.append(cd)
            cat_iou.append(iou)
            cat_miss.append(miss)
            cat_red.append(red)
            cat_f1.append(f1)
            names.append(name)
        cat_cds.append(np.array(cat_cd).mean())
        cat_ious.append(np.array(cat_iou).mean())
        cat_misses.append(np.array(cat_miss).mean())
        cat_reds.append(np.array(cat_red).mean())
        cat_f1s.append(np.array(cat_f1).mean())
        sum_len += len(data_l)
    for cd in cat_cds:
        print (cd)
    for iou in cat_ious:
        print (iou)
    print ("instance_cd")
    print (sum_cd / sum_len)
    # print ("category cd")
    print (np.array(cat_cds).mean())  
    # print ("instance_iou")
    print (sum_iou / sum_len)
    # print ("category iou")
    print (np.array(cat_ious).mean())
    # print ("instance_miss")
    print (sum_miss / sum_len)
    # print ("category miss")
    print (np.array(cat_misses).mean())
    # print ("instance_red")
    print (sum_red / sum_len)
    # print ("category red")
    print (np.array(cat_reds).mean())
    # print ("instance_f1")
    print (sum_f1 / sum_len)
    # print ("category f1")
    print (np.array(cat_f1s).mean())

def print_results_scannet(ious):
    cats_iou = {}
    for name, iou in ious.items():
        cat = name.split('/')[-1].split('_')[0]
        if cat in cats_iou:
            cats_iou[cat].append(iou)
        else:
            cats_iou[cat] = [iou]
    sum_iou = 0
    sum_miss = 0
    sum_red = 0
    sum_f1 = 0
    sum_len = 0
    cat_ious = []
    cat_misses = []
    cat_reds = []
    cat_f1s = []
    good = []
    bad = []
    for cat, data_l in cats_iou.items():
        cat_iou = []
        cat_miss = []
        cat_red = []
        cat_f1 = []
        names = []
        for iou, miss, red, f1, name in data_l:
            sum_iou += iou
            sum_miss += miss
            sum_red += red
            sum_f1 += f1
            cat_iou.append(iou)
            cat_miss.append(miss)
            cat_red.append(red)
            cat_f1.append(f1)
            names.append(name)
        cat_ious.append(np.array(cat_iou).mean())
        cat_misses.append(np.array(cat_miss).mean())
        cat_reds.append(np.array(cat_red).mean())
        cat_f1s.append(np.array(cat_f1).mean())
        sum_len += len(data_l)
        bad_idx = cat_iou.index(min(cat_iou))
        bad.append(names[bad_idx])
        good_idx = cat_iou.index(max(cat_iou))
        good.append(names[good_idx])

    for iou in cat_ious:
        print (iou)
    # print ("instance_iou")
    print (sum_iou / sum_len)
    # print ("category iou")
    print (np.array(cat_ious).mean())
    # print ("instance_miss")
    print (sum_miss / sum_len)
    # print ("category miss")
    print (np.array(cat_misses).mean())
    # print ("instance_red")
    print (sum_red / sum_len)
    # print ("category red")
    print (np.array(cat_reds).mean())
    # print ("instance_f1")
    print (sum_f1 / sum_len)
    # print ("category f1")
    print (np.array(cat_f1s).mean())
    print (sum_len)
    print ('good')
    names = []
    for name in good:
        scene, mask = name.split('/')[-2:]
        cat, model, idx = mask.split('_')[:3]
        names.append(os.path.join(scene, cat+'_'+model+'_'+idx+'_mask.npz'))
    print (names)
    print ('bad')
    names = []
    for name in bad:
        scene, mask = name.split('/')[-2:]
        cat, model, idx = mask.split('_')[:3]
        names.append(os.path.join(scene, cat+'_'+model+'_'+idx+'_mask.npz'))
    print (names)

def main():
    """
    Evaluate chamfer distance on prediction
    """
    args = parse_arguments().parse_args()
    evaluator = Evaluation(args)
    if args.dataset == "shapenet":
        eval_res = evaluator.evaluate_shapenet()
        print_results(eval_res)
    if args.dataset == "scannet":
        eval_res = evaluator.evaluate_scannt()
        print_results_scannet(eval_res)

if __name__ == "__main__":
    main()
