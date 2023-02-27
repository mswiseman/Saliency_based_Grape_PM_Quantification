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
                
                
#Text below provides parameter explanations and usage directions

#To use: 

#1. Change parameters if desired
#2. Save
#3. Ensure all zeros in front of image file names have been removed by running `bash ../common/remove_zero.sh`
#4. Open terminal
#5. bash leaf_correlation.sh

#General Notes for above parameters:

#--model_type
    #The type of deep learning model to train. Choices include: GoogleNet, ResNet, SqueezeNet, DenseNet, VGG, Inception3, Alexnet.

#--model_path
#    - This is essentially the root path
#    - on PC, looks like this: /c/Users/michele.wiseman/Desktop/Saliency_based_Grape_PM_Quantification-main
#    - on OSx, looks like this: 

#--dataset_path
#    - Where the image data is housed, e.g. /d/Stacked/Deposition_Study

#--pretrained
#    - If you are using a pretrained model (e.g. Inception3, DeepLab, etc., you should have the --pretrained flag)

#--epoch
#    - The specific epoch of the model you're loading (e.g. Inception3_model_ep095 would be 95)

#--threshold

#--cuda
#    - Include this tag if you have a CUDA enabled GPU
#    - Can check by running (in python after import torch): print(torch.cuda.is_available())
#    - More info here: https://developer.nvidia.com/cuda-toolkit

#--cuda_id
#    - The ID of your CUDA-endabled GPU (if using one) in case you have more than one
#    - Default is 0
#    - To determine, run: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

#--timestamp
#    - Timestamp on the specific model being tested (e.g. Feb06_09-26-58_2023) 

#--group
#    - Optional, but provides architecture for better subdirectory organization
#    - E.g. if a specific mapping population, specify that here (--group comet_64305M)

#--img_folder 
#    - Name of your image folder (e.g. 2-5-2023_6dpi) 

#--platform
#    - BlackBird or PMbot.
#    - Changing this will slightly alter masking thresholds.
