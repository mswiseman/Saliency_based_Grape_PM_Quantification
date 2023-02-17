#!/bin/bash
# Before running, be sure to remove zeros in your image file ids using the ../common/remove_zeros.sh script

time python3 ../leaf_correlation.py \
                --model_type VGG                                           \
                --model_path /c/Users/michele.wiseman/Desktop/blackbird/   \
                --dataset_path /d/Stacked/Deposition_Study                 \
                --pretrained                                               \
                --loading_epoch 281                                        \
                --threshold 0.2                                            \
                --cuda                                                     \
                --cuda_id 0                                                \
                --timestamp Feb07_00-10-08_2023                            \
                --group deposition_study                                   \
                --img_folder 2-5-2023_6dpi                                 \
                --platform BlackBird

"""
Text below provides parameter explanations and usage directions

To use: 

1. Change parameters if desired
2. Save
3. Ensure all zeros in front of image file names have been removed by running `bash ../common/remove_zero.sh`
4. Open terminal
5. bash leaf_correlation.sh

"""

"""

General Notes for above parameters:

--model_type
    - The type of deep learning model to train. Choices include:

        - GoogleNet: GoogleNet is a 22-layer deep Convolutional Neural Network (CNN) introduced in 2014. It uses the inception module, which 
            concatenates multiple filters with different sizes to form a more diverse representation of the image. The final layer of GoogleNet
            is a softmax classifier, used for image classification. https://arxiv.org/abs/1409.4842

        - ResNet: ResNet stands for Residual Network, introduced in 2015. ResNets are deep neural networks with hundreds or even thousands of 
            layers, which were difficult to train before. The main idea behind ResNet is to use residual connections, where the input from one layer 
            is added to the output of another layer. This allows the network to better preserve information and make the training of deep networks 
            easier. https://arxiv.org/abs/1512.03385

        - SqueezeNet: SqueezeNet is a light-weight CNN introduced in 2016, designed to have fewer parameters while still achieving high accuracy 
            on image classification tasks. It uses the Fire module, which concatenates 1x1 and 3x3 filters, to reduce the number of parameters.
            https://arxiv.org/abs/1602.07360

        - DenseNet: DenseNet is a type of CNN introduced in 2016, which uses dense connections instead of residual connections. In a dense network,
            each layer is connected to all previous layers, leading to a dense flow of information through the network. This architecture helps 
            reduce the number of parameters, improve feature reuse, and promote feature fusion. https://arxiv.org/abs/1608.06993

        - VGG: VGG is a very deep CNN introduced in 2014, which uses a simple architecture with only 3x3 convolutions and max-pooling. It has a 
            large number of parameters, but it still manages to achieve good results on image classification tasks. https://arxiv.org/abs/1409.1556

        - AlexNet: AlexNet is an 8-layer deep CNN introduced in 2012, which was one of the first deep networks to win the ImageNet Large Scale 
            Visual Recognition Challenge (ILSVRC). AlexNet uses the ReLU activation function, max-pooling, and dropout to prevent overfitting.
            https://dl.acm.org/doi/10.1145/3065386# --timestamp is the model timestamp, e.g. 'Feb05_22-39-19_2023'

--model_path
    - This is essentially the root path
    - on PC, looks like this: /c/Users/michele.wiseman/Desktop/blackbird

--dataset_path
    - Where the image data is housed, e.g. /d/Stacked/Deposition_Study

--pretrained
    - If you are using a pretrained model (e.g. Inception3, DeepLab, etc., you should have the --pretrained flag)

--epoch
    - The specific epoch of the model you're loading (e.g. Inception3_model_ep095 would be 95)

--threshold

--cuda
    - Include this tag if you have a CUDA enabled GPU
    - Can check by running (in python after import torch): print(torch.cuda.is_available())
    - More info here: https://developer.nvidia.com/cuda-toolkit

--cuda_id
    - The ID of your CUDA-endabled GPU (if using one) in case you have more than one
    - Default is 0
    - To determine, run: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

--timestamp
    - Timestamp on the specific model being tested (e.g. Feb06_09-26-58_2023) 

--group
    - Optional, but provides architecture for better subdirectory organization
    - E.g. if a specific mapping population, specify that here (--group comet_64305M)

--img_folder 
    - Name of your image folder (e.g. 2-5-2023_6dpi) 

--platform
    - BlackBird or PMbot.
    - Changing this will slightly alter masking thresholds.

"""
