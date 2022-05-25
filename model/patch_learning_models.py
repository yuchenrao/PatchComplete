
import numpy as np
import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock_large(nn.Module):
    # BasicBlock places the stride for downsampling at 3x3 convolution for nn.conv3d
    # according to Bottleneck in torchvision.resnet 
    # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    def __init__(self,
                 mode: str,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int=1,
                 padding: int=1,
                 output_padding: int=1,
                 use_batchnorm: bool=True,
                 leaky: bool=False):
        super(BasicBlock_large, self).__init__()

        if mode == 'Encoder':
            self._conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        elif mode == 'Decoder':
            self._conv1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        else:
            print ("Wrong mode, please enter 'Encoder' or 'Decoder'.")
            return
        self._conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self._conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if leaky:
            self._relu = nn.LeakyReLU(0.2) 
        else:    
            self._relu = nn.ReLU(inplace=True)
        self._use_batchnorm = use_batchnorm
        if self._use_batchnorm:
            self._bn_1 = nn.BatchNorm3d(out_channels)
            self._bn_2 = nn.BatchNorm3d(out_channels)
            self._bn_3 = nn.BatchNorm3d(out_channels)
        else:
            self._conv1 = nn.utils.weight_norm(self._conv1, name='weight')
            self._conv2 = nn.utils.weight_norm(self._conv2, name='weight')
            self._conv3 = nn.utils.weight_norm(self._conv3, name='weight')
           
    def forward(self, x):
        out = None
        identity = None

        if self._use_batchnorm:
            out = self._conv1(x)
            out = self._bn_1(out)
            out = self._relu(out)

            identity = out
            out = self._conv2(out)
            out = self._bn_2(out)
            out = self._relu(out)
            out = self._conv3(out)
            out = self._bn_3(out)
        
        else:
            out = self._conv1(x)
            out = self._relu(out)

            identity = out
            out = self._conv2(out)
            out = self._relu(out)
            out = self._conv2(out)

        out += identity
        out = self._relu(out)

        return out

class PatchLearningModel_fewshot_priors_3_encoder(nn.Module):
    def __init__(self, use_batchnorm, device, channel_num, patch_res, truncation=3, input_res=32):
        super(PatchLearningModel_fewshot_priors_3_encoder, self).__init__()
        """
        :param use_batchnorm: whether use batchnorm or not
        :type use_batchnorm: bool
        :param device: divice_name
        :type device: str
        :param channel_num: the number of channel for input feature
        :type channel_num: int
        :param patch_res: patch resolution
        :type patch_res: int
        :param truncation: truncation value
        :type truncation: float
        :param input_res: input resolution
        :type input_res: int
        """
        self._device = device
        self._channel_num = channel_num
        self._patch_res = patch_res
        self._patch_num_edge = int(input_res / self._patch_res)
        # prior path
        self._priors_path = 'priors'
        self._truncation = truncation

        if self._patch_res == 32:
            self._encoder_input_1 = BasicBlock_large('Encoder', 1, int(self._channel_num / 2), 3, 2, use_batchnorm=False)
            self._encoder_input_2 = BasicBlock_large('Encoder', int(self._channel_num / 2), self._channel_num, 3, 2, use_batchnorm=use_batchnorm)
            self._encoder_input_3 = BasicBlock_large('Encoder', self._channel_num, self._channel_num * 2, 4, 4, use_batchnorm=use_batchnorm)
            self._encoder_priors_1 = BasicBlock_large('Encoder', 1, int(self._channel_num / 2), 3, 2, use_batchnorm=False)
            self._encoder_priors_2 = BasicBlock_large('Encoder', int(self._channel_num / 2), self._channel_num, 3, 2, use_batchnorm=use_batchnorm)
            self._encoder_priors_3 = BasicBlock_large('Encoder', self._channel_num, self._channel_num * 2, 4, 4, use_batchnorm=use_batchnorm)
            # linear layers for weights
            self._weights_1 = nn.Linear(112, 112, bias=False)
            self._weights_bn_1 = nn.BatchNorm1d(112)
            self._weights_2 = nn.Linear(112, 112, bias=False)
            self._weights_bn_2 = nn.BatchNorm1d(112)
        if self._patch_res == 8:
            self._encoder_input_1 = BasicBlock_large('Encoder', 1, int(self._channel_num / 2), 3, 2, use_batchnorm=False)
            self._encoder_input_2 = BasicBlock_large('Encoder', int(self._channel_num / 2), self._channel_num, 3, 2, use_batchnorm=use_batchnorm)
            self._encoder_input_3 = BasicBlock_large('Encoder', self._channel_num, self._channel_num * 2, 3, 2, use_batchnorm=use_batchnorm)
            self._encoder_priors_1 = BasicBlock_large('Encoder', 1, int(self._channel_num / 2), 3, 2, use_batchnorm=False)
            self._encoder_priors_2 = BasicBlock_large('Encoder', int(self._channel_num / 2), self._channel_num, 3, 2, use_batchnorm=use_batchnorm) 
            self._encoder_priors_3 = BasicBlock_large('Encoder', self._channel_num, self._channel_num * 2, 3, 2, use_batchnorm=use_batchnorm) 
            # linear layers for weights
            self._weights_1 = nn.Linear(112*64, 112*64, bias=False)
            self._weights_bn_1 = nn.BatchNorm1d(112*64)
            self._weights_2 = nn.Linear(112*64, 112*64, bias=False) 
            self._weights_bn_2 = nn.BatchNorm1d(112*64)
        if self._patch_res == 4:
            self._encoder_input_1 = BasicBlock_large('Encoder', 1, int(self._channel_num / 2), 3, 2, use_batchnorm=False)
            self._encoder_input_2 = BasicBlock_large('Encoder', int(self._channel_num / 2), self._channel_num, 3, 2, use_batchnorm=use_batchnorm)
            self._encoder_input_3 = BasicBlock_large('Encoder', self._channel_num, self._channel_num * 2, 3, 1, padding=1, use_batchnorm=use_batchnorm)
            self._encoder_priors_1 = BasicBlock_large('Encoder', 1, int(self._channel_num / 2), 3, 2, use_batchnorm=False)
            self._encoder_priors_2 = BasicBlock_large('Encoder', int(self._channel_num / 2), self._channel_num, 3, 2, use_batchnorm=use_batchnorm)
            self._encoder_priors_3 = BasicBlock_large('Encoder', self._channel_num, self._channel_num * 2, 3, 1, padding=1, use_batchnorm=use_batchnorm)
        self._softmax = nn.Softmax(1)
        self._relu = nn.ReLU(inplace=False)

        priors = self._get_data()
        priors = torch.from_numpy(np.array(priors)).unsqueeze(1)
        self._codebook = None
        self._codebook = nn.Parameter(priors.float().to(self._device), requires_grad=True)

    def _get_data(self):
        """
        This function reads all the shape priors for training classes generated by using ShapeNet GT models for training samples
        """
        priors = []
        for proir_file in glob.iglob(os.path.join(self._priors_path, "*.npy")):
            with open(proir_file, 'rb') as data:
                prior = np.load(data)
                prior[np.where(prior > self._truncation)] = self._truncation
                prior[np.where(prior < -1* self._truncation)] = -1 * self._truncation
                priors.append(prior)
        return priors

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        # get input features
        instance_features = self._encoder_input_1(inputs) 
        instance_features = self._encoder_input_2(instance_features)
        instance_features = self._encoder_input_3(instance_features)
        patch_features = torch.flatten(instance_features, start_dim=2).transpose(1, 2)
        if self._patch_res == 32:
            patch_features = torch.flatten(patch_features, start_dim=1)
        else:
            patch_features = torch.flatten(patch_features, start_dim=0, end_dim=1)

        # get prior features
        priors = torch.clamp(self._codebook, min=-1*self._truncation, max=self._truncation)
        prior_features = self._encoder_priors_1(priors)
        prior_features = self._encoder_priors_2(prior_features)
        prior_features = self._encoder_priors_3(prior_features)
        prior_features = torch.flatten(prior_features, start_dim=2).transpose(1, 2)
        if self._patch_res == 32:
            prior_features = torch.flatten(prior_features, start_dim=1)
        else:
            prior_features = torch.flatten(prior_features, start_dim=0, end_dim=1)
        
        # get attention scores
        patch_features = patch_features / self._channel_num
        learned_weights = torch.matmul(patch_features, prior_features.transpose(1, 0))
        if self._patch_res != 4:
            learned_weights = self._relu(self._weights_bn_1(self._weights_1(learned_weights)))
            learned_weights = self._relu(self._weights_bn_2(self._weights_2(learned_weights)))
        learned_weights = self._softmax(learned_weights)

        # get chuncked prior patches
        learned_patches = self._codebook.unfold(2, self._patch_res, self._patch_res).unfold(3, self._patch_res, self._patch_res).unfold(4, self._patch_res, self._patch_res)
        learned_patches = torch.flatten(learned_patches, start_dim=2, end_dim=4).transpose(1, 2)
        learned_patches = torch.flatten(learned_patches, start_dim=0, end_dim=1) 
        
        # sum weighted prior patches and reunion
        final_shape = torch.matmul(learned_weights, torch.flatten(learned_patches, start_dim=1))
        final_shape = final_shape.reshape(batch_size, self._patch_num_edge ** 3, self._patch_res, self._patch_res, self._patch_res)
        final_shape = final_shape.view(batch_size, 1, self._patch_num_edge, self._patch_num_edge, self._patch_num_edge, self._patch_res, self._patch_res, self._patch_res) #[1, 1, 4, 4, 4, 8, 8, 8]
        final_shape = final_shape.permute(0, 1, 2, 5, 3, 6, 4, 7).reshape(batch_size, 1, self._patch_res * self._patch_num_edge, self._patch_res * self._patch_num_edge, self._patch_res * self._patch_num_edge) #[1, 1, 32, 32, 32]
        final_shape = torch.clamp(final_shape, min=-1 * self._truncation, max=self._truncation)
        
        return final_shape, prior_features, instance_features, learned_weights

class ShapeLearningModel_codebook_learning_end_to_end_flatten(nn.Module):
    def __init__(self, use_batchnorm, device, channel_num, model_shape, model_8, model_4):
        super(ShapeLearningModel_codebook_learning_end_to_end_flatten, self).__init__()
        """
        :param use_batchnorm: whether use batchnorm or not
        :type use_batchnorm: bool
        :param device: divice_name
        :type device: str
        :param channel_num: the number of channel for input feature
        :type channel_num: int
        :param model_shape: patch_learning model for res_32 
        :type model_shape: patch_learning_models.PatchLearningModel_fewshot_priors_3_encoder
        :param model_8: patch_learning model for res_8
        :type model_8: patch_learning_models.PatchLearningModel_fewshot_priors_3_encoder
        :param model_4: patch_learning model for res_4
        :type model_smodel_4hape: patch_learning_models.PatchLearningModel_fewshot_priors_3_encoder
        """
        self._device = device
        self._channel_num = channel_num
        self._model_shape = model_shape
        self._model_8 = model_8
        self._model_4 = model_4


        self._decoder_1 = BasicBlock_large('Decoder', self._channel_num*2, self._channel_num, 3, 2, use_batchnorm=use_batchnorm)
        self._decoder_2 = BasicBlock_large('Decoder', self._channel_num*3, self._channel_num, 3, 2, use_batchnorm=use_batchnorm) 
        self._decoder_3 = BasicBlock_large('Decoder', self._channel_num*3, self._channel_num, 4, 4, output_padding=2, use_batchnorm=use_batchnorm)
        self._conv_last = nn.Conv3d(self._channel_num, 1, 3, 1, 1)
        self._softmax = nn.Softmax(1)

    def forward(self, inputs):
        """
        This function gets the shape priors' distributions for inputs,
        and returned learned shape priors for imputs
        """
        batch_size = inputs.shape[0]

        _, latent_shape, input_feature_shape, learned_weights_shape = self._model_shape(inputs)
        _, latent_8, input_feature_8, learned_weights_8 = self._model_8(inputs)
        _, latent_4, input_feature_4, learned_weights_4 = self._model_4(inputs)

        # decoder for res_32
        final_shape = torch.matmul(learned_weights_shape, latent_shape)
        final_shape = final_shape.view(batch_size, 8, self._channel_num).transpose(1, 2).view(batch_size, self._channel_num, 2, 2, 2)
        final_shape = torch.cat((final_shape, input_feature_shape), dim=1)
        final_shape = self._decoder_1(final_shape)

        # decoder for res_8
        final_shape_8 = torch.matmul(learned_weights_8, latent_8)
        final_shape_8 = final_shape_8.reshape(batch_size, 4 ** 3, self._channel_num).transpose(1, 2)
        final_shape_8 = final_shape_8.view(batch_size, self._channel_num, 4, 4, 4)
        final_shape = torch.cat([final_shape, final_shape_8], dim=1)
        final_shape = torch.cat((final_shape, input_feature_8), dim=1)
        final_shape = self._decoder_2(final_shape)

        # decoder for res_4
        final_shape_4 = torch.matmul(learned_weights_4, latent_4)
        final_shape_4 = final_shape_4.reshape(batch_size, 8 ** 3, self._channel_num).transpose(1, 2)
        final_shape_4 = final_shape_4.view(batch_size, self._channel_num, 8, 8, 8)
        final_shape = torch.cat([final_shape, final_shape_4], dim=1)

        # final conv
        output = torch.cat((final_shape, input_feature_4), dim=1)
        output = self._decoder_3(output)

        output = self._conv_last(output)
        
        return output, latent_shape, latent_8, latent_4
