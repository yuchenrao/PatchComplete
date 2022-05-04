import os
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model.patch_learning_dataset as patch_learning_dataset
import model.patch_learning_models as patch_learning_model

def generate(device, model, test_dataloader, output_path):
    """
    This function evaluate the trained models

    :param device: which device is used for training
    :type device: str
    :param model: trained model
    :type model: torch.nn.Module
    :param test_dataloader: dataloader for test dataset
    :type test_dataloader: torch.utils.data.DataLoader
    :param output_path: path to save prediction
    :type output_path: str
    """
    model.eval()
    dataset = output_path.split('/')[-1]
    with torch.no_grad():
        for inputs, _, _, names in test_dataloader:
            for i, mask_name in enumerate(names):
                inputs = inputs.to(device)
                shape_priors, _, _, _ = model(inputs)
                shape_priors = shape_priors.cpu().numpy()
                if dataset=='shapenet':
                    folder_name = mask_name.split('/')[-3]
                    folder_path = os.path.join(output_path, folder_name)
                    os.makedirs(folder_path, exist_ok=True)
                    print (folder_path)
                    folder_name = mask_name.split('/')[-2]
                    folder_path = os.path.join(folder_path, folder_name)
                    os.makedirs(folder_path, exist_ok=True)
                    print (folder_path)
                else:
                    folder_name = mask_name.split('/')[-2]
                    folder_path = os.path.join(output_path, folder_name)
                    os.makedirs(folder_path, exist_ok=True)
                file_name = mask_name.split('/')[-1][:-4] + '_pred.npz'
                file_path = os.path.join(folder_path, file_name)
                np.savez(file_path, predicted_voxels=shape_priors[i][0])
    return

def parse_arguments():
    """
    Generates a command line parser that supports all arguments used by the
    tool.

    :return: Parser with all arguments.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    # optimal parameters
    parser.add_argument("--test_file",
                        help="path to save the test dataset",
                        default="test.txt",
                        required=True,
                        type=str)
    parser.add_argument("--data_path",
                        help="path for getting data",
                        default="data_samples",
                        required=True,
                        type=str)
    parser.add_argument("--model_name",
                        help="model name for prediction",
                        default="multi_res.pt",
                        required=True,
                        type=str)
    parser.add_argument("--dataset",
                        help="dataset",
                        required=True,
                        type=str)
    parser.add_argument("--model_stage",
                        help="model stage (patch_learning, multi_res)",
                        required=True,
                        type=str)
    parser.add_argument("--model_path",
                        help="path to save the trained model",
                        default="trained_models",
                        type=str)
    parser.add_argument("--output_path",
                        help="path to save predictions",
                        default="output",
                        type=str)
    parser.add_argument("--batch_size",
                        help="batch size for training",
                        default=32,
                        type=int)
    parser.add_argument("--truncation",
                        help="truncation values for sdf",
                        default=3.0,
                        type=float) 
    parser.add_argument("--channel_num",
                        help="number of channels for learning",
                        default=128,
                        type=int)
    parser.add_argument("--patch_res",
                        help="patch resoluion for patch learning stage",
                        default=32,
                        type=int)
    parser.add_argument('--no_batchnorm', dest='no_batchnorm', action='store_false')
    parser.set_defaults(feature=False)

    return parser

def main():
    """
    generates binary masks for training
    """
    args = parse_arguments().parse_args()
    if args.dataset == 'shapenet':
        test_set = patch_learning_dataset.ShapenetDataset(args.test_file, args.data_path, truncation=args.truncation)
    elif args.dataset == 'scannet':
        test_set = patch_learning_dataset.ScannetDataset(args.test_file, args.data_path, truncation=args.truncation, use_bbox=True)
    # create dataset
    dataloaders = {
        'test': DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    }
    print ("Finish data loading")

    # init model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_stage == 'patch_learning':
        model = patch_learning_model.PatchLearningModel_fewshot_priors_3_encoder(args.no_batchnorm, device, args.channel_num, args.patch_res, truncation=args.truncation)
    elif args.model_stage == 'multi_res':
        model_shape = patch_learning_model.PatchLearningModel_fewshot_priors_3_encoder(args.no_batchnorm, device, args.channel_num, 32, truncation=args.truncation)
        model_shape.load_state_dict(torch.load(os.path.join(args.model_path, "patch_learning_res_32.pt"))) 
        model_shape = model_shape.to(device)
        model_8 = patch_learning_model.PatchLearningModel_fewshot_priors_3_encoder(args.no_batchnorm, device, args.channel_num, 8, truncation=args.truncation)
        model_8.load_state_dict(torch.load(os.path.join(args.model_path, "patch_learning_res_8.pt"))) 
        model_8 = model_8.to(device)
        model_4 = patch_learning_model.PatchLearningModel_fewshot_priors_3_encoder(args.no_batchnorm, device, args.channel_num, 4, truncation=args.truncation)
        model_4.load_state_dict(torch.load(os.path.join(args.model_path, "patch_learning_res_4.pt")))
        model_4 = model_4.to(device)
        model = patch_learning_model.ShapeLearningModel_codebook_learning_end_to_end_flatten(False, device, int(args.channel_num*2), model_shape, model_8, model_4)
    model_name = args.model_name
    model_path = os.path.join(args.model_path, model_name)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    print (sum(p.numel() for p in model.parameters() if p.requires_grad))
    print (model)

    # generation
    print ("Start generation--------")
    os.makedirs(args.output_path, exist_ok=True)
    output_path = os.path.join(args.output_path, args.dataset)
    os.makedirs(output_path, exist_ok=True)
    generate(device, model, dataloaders['test'], output_path)

if __name__ == "__main__":
    main()