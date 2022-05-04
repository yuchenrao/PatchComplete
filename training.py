import os
import time
import numpy as np
import argparse
import time
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("tboard/training_curve")

import model.patch_learning_dataset as patch_learning_dataset
import model.patch_learning_models as patch_learning_model

def calc_loss_l1_weighted(shape_priors, target_sdf, metrics, name):
    criterion_l1 = nn.SmoothL1Loss(reduction='none')
    loss_l1_ori = criterion_l1(shape_priors, target_sdf)
    target_mask = np.zeros(target_sdf.shape)
    target_mask[np.where(target_sdf.cpu().detach().numpy() >= 1e-10)] = 1 # opposite but better
    pred_mask = np.zeros(shape_priors.shape)
    pred_mask[np.where(shape_priors.cpu().detach().numpy() >= 1e-10)] = 1
    false_pos_mask = np.zeros(shape_priors.shape)
    false_neg_mask = np.zeros(shape_priors.shape)
    false_pos_mask[np.where((pred_mask==1) & (target_mask==0))] = 1
    false_pos_mask_cuda = torch.from_numpy(false_pos_mask).cuda()
    false_neg_mask[np.where((pred_mask==0) & (target_mask==1))] = 1
    false_neg_mask_cuda = torch.from_numpy(false_neg_mask).cuda()
    loss_l1 = torch.mean(false_pos_mask_cuda * loss_l1_ori * 5 + false_neg_mask_cuda * loss_l1_ori * 3 + (1 - false_neg_mask_cuda - false_pos_mask_cuda) * loss_l1_ori) # weight false positiva / false negative

    loss_l1_data = loss_l1.data.cpu().numpy()
    metrics['loss'+name] += loss_l1_data * target_sdf.size(0)

    iou_sum = 0
    for idx, p in enumerate(shape_priors):
        # calculate IOU
        new_p = np.zeros(p[0].shape)
        new_p[np.where(p[0].cpu().detach().numpy() <= 0)] = 1
        new_mask = np.zeros(target_sdf[idx][0].shape)
        new_mask[np.where(target_sdf[idx][0].cpu().detach().numpy() <= 0)] = 1
        result = new_p + new_mask
        iou_sum += (np.sum(np.array(result) == 2) / np.sum(np.array(result) >= 1))
    metrics['iou'+name] += iou_sum

    return loss_l1

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))    

def jitter(src, tgt, jitter, truncation, tgtTruncation):
    dst = torch.full(src.shape, truncation)
    tgtdst = torch.full(tgt.shape, tgtTruncation) 
    for idx in range(src.shape[0]):
        i = random.randint(-jitter, jitter)
        j = random.randint(-jitter, jitter)
        k = random.randint(-jitter, jitter)
        if i >= 0:
            xidx = [i, dst.shape[-1] - 1, 0, dst.shape[-1] - i - 1]
        if i < 0:
            xidx = [0, dst.shape[-1] + i - 1, -i, dst.shape[-1] - 1]
        if j >= 0:
            yidx = [j, dst.shape[-1] - 1, 0, dst.shape[-1] - j - 1]
        if j < 0:
            yidx = [0, dst.shape[-1] + j- 1, -j, dst.shape[-1] - 1]
        if k >= 0:
            zidx = [k, dst.shape[-1] - 1, 0, dst.shape[-1] - k - 1]
        if k < 0:
            zidx = [0, dst.shape[-1] + k - 1, -k, dst.shape[-1] - 1]
 
        dst[idx, :, xidx[0] : xidx[1], yidx[0] : yidx[1], zidx[0] : zidx[1]] = src[idx, :, xidx[2] : xidx[3], yidx[2] : yidx[3], zidx[2] : zidx[3]]
        tgtdst[idx, :, xidx[0] : xidx[1], yidx[0] : yidx[1], zidx[0] : zidx[1]] = tgt[idx, :, xidx[2] : xidx[3], yidx[2] : yidx[3], zidx[2] : zidx[3]]
    return dst, tgtdst

def transpose_axis(inputs, labels, trans_list):
    pick = random.randint(0, 1)
    axis_order = np.array(trans_list[pick]) + 2
    # transpose
    inputs = inputs.permute(0, 1, axis_order[0], axis_order[1], axis_order[2])
    labels = labels.permute(0, 1, axis_order[0], axis_order[1], axis_order[2]) 

    return inputs, labels

def mirror(inputs, labels):
    pick = random.randint(0, 3)
    if pick==0:
        return inputs, labels
    elif pick==1:
        return torch.flip(inputs, [2]), torch.flip(labels, [2])
    elif pick==2:
        return torch.flip(inputs, [4]), torch.flip(labels, [4])
    else:
        return torch.flip(inputs, [2, 4]), torch.flip(labels, [2, 4])

def add_walls(inputs, bbox):
    pick = random.randint(0, 6)
    if pick == 0:
        inputs[np.array(np.linspace(0, inputs.shape[0] -1, num=inputs.shape[0]), dtype=int), :, np.array(bbox[:, :, 0, 0]).flatten(), :, :] = -0.5
    elif pick == 1:
        inputs[np.array(np.linspace(0, inputs.shape[0] -1, num=inputs.shape[0]), dtype=int), :, np.array(bbox[:, :, 0, 1]).flatten(), :, :] = -0.5
    elif pick == 2:
        inputs[np.array(np.linspace(0, inputs.shape[0] -1, num=inputs.shape[0]), dtype=int), :, np.array(bbox[:, :, 1, 0]).flatten(), :, :] = -0.5
    elif pick == 3:
        inputs[np.array(np.linspace(0, inputs.shape[0] -1, num=inputs.shape[0]), dtype=int), :, np.array(bbox[:, :, 1, 1]).flatten(), :, :] = -0.5
    elif pick == 4:
        inputs[np.array(np.linspace(0, inputs.shape[0] -1, num=inputs.shape[0]), dtype=int), :, np.array(bbox[:, :, 2, 0]).flatten(), :, :] = -0.5
    elif pick == 5:
        inputs[np.array(np.linspace(0, inputs.shape[0] -1, num=inputs.shape[0]), dtype=int), :, np.array(bbox[:, :, 2, 1]).flatten(), :, :] = -0.5
    return inputs

def mask_generation(inputs, mask_value):
    mask = np.ones(32*32*32, dtype=int)
    mask[:3000] = 0
    np.random.shuffle(mask)
    mask = torch.from_numpy(mask)
    mask = mask.reshape((32,32,32))
    masked_inputs = inputs * mask + mask_value * (1 - mask)
    return masked_inputs

def train(device, model, dataloaders, num_epochs, model_stage, output_path, truncation, lr, no_walls_aug, patch_res=0):
    """
    This function trains a model based on the training and validation datasets

    :param device: which device is used for training
    :param type: str
    :param model: trainning model
    :type model: torch.nn.Module
    :param dataloaders: dataloader for training and validation dataset
    :type dataloaders: torch.utils.data.DataLoader
    :param num_epochs: the number of epoches
    :param num_epochs: int
    :param model_stage: model stage: patch_learning, multi_res, fine_tune
    :type model_ref: str
    :param output_path: output path for saving models
    :type output_path: str
    :param lr: learning rate
    :type lr: float
    :param no_walls_aug: whether to add walls for data augmentation or not
    :type no_walls_aug: bool
    :param patch_res: patch resolution for patch_learning stage
    :type patch_res: float
    :return model: trained model
    :rtype: torch.nn.Module
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)#, weight_decay=0.0001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5) 

    # fix parameters for training
    for name, p in model.named_parameters():
        if model_stage == "multires":
            if '_decoder' in name or '_conv_last' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        elif model_stage == "fine_tune":
            if '_encoder_input' in name :
                p.requires_grad = True
            else:
                p.requires_grad = False

    # iterate epoches
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val_trained', 'val_novel']:
            print (phase)
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()
            # start process
            metrics = defaultdict(float)
            epoch_samples = 0
            
            for inputs, labels, bbox, _ in dataloaders[phase]:
                if 
                if phase == 'train':
                    # data augmentation
                    if no_walls_aug is False:
                        # only for shapenet pretrain
                        inputs = add_walls(inputs, bbox)
                    inputs, labels = mirror(inputs, labels)
                    inputs, labels = transpose_axis(inputs, labels, [[0, 1, 2],[2, 1, 0]])
                    inputs, labels = jitter(inputs, labels, 2, truncation, truncation) 
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)
                # zero the parameter gradients
                with torch.set_grad_enabled(phase == 'train'):
                    shape_priors_generated, _, _, _ = model(inputs)
                    loss = calc_loss_l1_weighted(shape_priors_generated, labels, metrics, '')
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)
            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            epoch_iou = metrics['iou'] / epoch_samples
            
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                writer.add_scalar("Loss/train", epoch_loss, epoch)
                writer.add_scalar("iou/train", epoch_iou, epoch)
            elif phase == 'val_trained':
                writer.add_scalar("Loss/val_trained", epoch_loss, epoch)
                writer.add_scalar("iou/val_trained", epoch_iou, epoch)
            elif phase == 'val_novel':
                writer.add_scalar("Loss/val_novel", epoch_loss, epoch)
                writer.add_scalar("iou/val_novel", epoch_iou, epoch)
        
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        exp_lr_scheduler.step()
        # save model
        if (epoch + 1) % 20 == 0:
            if model_stage == "patch_learning":
                model_name = model_stage + "_res_" + str(patch_res) + "_epoch_" + str(epoch) + ".pt"
            else:
                model_name = model_stage + "_epoch_" + str(epoch) + ".pt"
            model_path = os.path.join(output_path, model_name)
            print ("save model {}".format(model_name))
            torch.save(model.state_dict(), model_path)
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
    parser.add_argument("--data_path",
                        help="path for getting data",
                        default="/Users/yuchenrao/",
                        required=True,
                        type=str)
    parser.add_argument("--dataset",
                        help="dataeset to be trained",
                        default="shapenet",
                        required=True,
                        type=str)
    parser.add_argument("--train_file",
                        help="path to save the trainning dataset",
                        default="./train.txt",
                        required=True,
                        type=str)
    parser.add_argument("--val_trained_file",
                        help="path to save the trained validation dataset",
                        default="./val_traiend.txt",
                        required=True,
                        type=str)
    parser.add_argument("--val_novel_file",
                        help="path to save the novel validation dataset",
                        default="./val_novel.txt",
                        required=True,
                        type=str)
    parser.add_argument("--output_path",
                        help="path to save the trained model",
                        default="trained_models",
                        type=str)
    parser.add_argument("--model_stage",
                        help="Specify the stage that we are training",
                        default="patch_learning",
                        required=True,
                        type=str)
    parser.add_argument("--batch_size",
                        help="batch size for training",
                        default=32,
                        type=int)
    parser.add_argument("--num_epochs",
                        help="the number of epochs",
                        default=80,
                        type=int)
    parser.add_argument("--truncation",
                        help="value for getting truncated udf",
                        default=3,
                        type=float)
    parser.add_argument("--patch_res",
                        help="patch resoluion for patch learning stage",
                        default=32,
                        type=int)
    parser.add_argument("--lr",
                        help="learning rate for training",
                        default=0.001,
                        type=float)
    parser.add_argument("--channel_num",
                        help="number of channels for learning",
                        default=128,
                        type=int)
    parser.add_argument("--gpu",
                        help="gpu index for training",
                        default=0,
                        type=int)
    parser.add_argument('--no_batchnorm', dest='no_batchnorm', action='store_false')
    parser.add_argument('--no_wall_aug', dest='no_wall_aug', action='store_false')
    parser.set_defaults(feature=False)

    return parser

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    """
    generates binary masks for training
    """
    args = parse_arguments().parse_args()
    # get file path
    if os.path.exists(args.data_path) is not True:
        print('Please use valid data_path')
        return
    if os.path.exists(args.train_file) is not True:
        print('Please use valid train file')
        return
    if os.path.exists(args.val_trained_file) is not True:
        print('Please use valid validation trained file')
        return
    if os.path.exists(args.val_novel_file) is not True:
        print('Please use valid validation novel file')
        return
    if os.path.exists(args.output_path) is not True:
        os.mkdir(args.output_path)
    # create dataset
    if args.dataset == 'shapenet':
        train_set = patch_learning_dataset.ShapenetDataset(args.train_file, args.data_path, truncation=args.truncation)
        val_trained_set = patch_learning_dataset.ShapenetDataset(args.val_trained_file, args.data_path, truncation=args.truncation)
        val_novel_set = patch_learning_dataset.ShapenetDataset(args.val_novel_file, args.data_path, truncation=args.truncation)
    elif args.dataset == 'scannet':
        train_set = patch_learning_dataset.ScannetDataset(args.train_file, args.data_path, truncation=args.truncation, use_bbox=True)
        val_trained_set = patch_learning_dataset.ScannetDataset(args.val_trained_file, args.data_path, truncation=args.truncation, use_bbox=True)
        val_novel_set = patch_learning_dataset.ScannetDataset(args.val_novel_file, args.data_path, truncation=args.truncation, use_bbox=True)
    else:
        print ("Please use valid datasets (shapenet, scannet)")
        return 
    print ("Finish data Loading")
    dataloaders = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker),
        'val_trained': DataLoader(val_trained_set, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker),
        'val_novel': DataLoader(val_novel_set, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker)
    }

    # set up device and model
    print("Init model")
    gpu = 'cuda:' + str(args.gpu)
    print ("using gpu: " + gpu)
    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
    print (device)
    manual_seed = 1
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if args.model_stage == 'patch_learning':
        model_refine = patch_learning_model.PatchLearningModel_fewshot_priors_3_encoder(args.no_batchnorm, device, args.channel_num, args.patch_res, truncation=args.truncation)
    elif args.model_stage == 'multi_res' or args.model_stage == 'fine_tune':
        model_shape = patch_learning_model.PatchLearningModel_fewshot_priors_3_encoder(args.no_batchnorm, device, args.channel_num, 32, truncation=args.truncation)
        model_shape.load_state_dict(torch.load(os.path.join(args.output_path, "patch_learning_res_32.pt"))) 
        model_shape = model_shape.to(device)
        model_8 = patch_learning_model.PatchLearningModel_fewshot_priors_3_encoder(args.no_batchnorm, device, args.channel_num, 8, truncation=args.truncation)
        model_8.load_state_dict(torch.load(os.path.join(args.output_path, "patch_learning_res_8.pt"))) 
        model_8 = model_8.to(device)
        model_4 = patch_learning_model.PatchLearningModel_fewshot_priors_3_encoder(args.no_batchnorm, device, args.channel_num, 4, truncation=args.truncation)
        model_4.load_state_dict(torch.load(os.path.join(args.output_path, "patch_learning_res_4.pt")))
        model_4 = model_4.to(device)
        model_refine = patch_learning_model.ShapeLearningModel_codebook_learning_end_to_end_flatten(False, device, int(args.channel_num*2), model_shape, model_8, model_4)
    elif args.model_stage == 'fine_tune':
        model_refine.load_state_dict(torch.load(args.output_path, "multi_res.pt")) 
    else:
        print ("Please use valid model stages (patch_learning, multi_res, fine_tune)")
        return

    model_refine = model_refine.to(device)
    print (model_refine)
    print (sum(p.numel() for p in model_refine.parameters() if p.requires_grad))

    # training
    print("Start training")
    start_time = time.time() 
    train(device, model_refine, dataloaders, args.num_epochs, args.model_stage, args.output_path, args.truncation, args.lr, args.no_wall_aug, args.patch_res)
    train_time = time.time() - start_time
    print ("training time: {}".format(train_time))

if __name__ == "__main__":
    main()
