# Road_extraction
Attention Unet and Deep Unet implementation for road extraction using multi-gpu model tensorflow

Several variations of Deep U-Net were tested with extra layers and extra convolutions. Nevertheless, the model that outperformed all of them was the Attention U-Net: Learning Where to Look for the Pancreas. I have added an extra tweak improving even further performance, switching the convolution blocks to the residual blocks

# TensorFlow Segmentation
TF segmentation models, U-Net, Attention Unet, Deep U-Net (All variations of U-Net)

Image Segmentation using neural networks (NNs), designed for extracting the road network from remote sensing imagery and it can be used in other applications labels every pixel in the image (Semantic segmentation) 

Details can be found in these papers:

* [Unet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)

## Attention U-Net extra module

![AU-Net](Images/aunet.png)


## Requirements
* Python 3.6
* CUDA 10.0
* TensorFlow 1.9
* Keras 2.0


## Modules
utils.py and helper.py 
functions for preprocessing data and saving it.


## Trainig model:
```
usage: mainGPU.py [-h] [--num_epochs NUM_EPOCHS] [--save SAVE] [--gpu GPU]
                  [--mode MODE] [--checkpoint CHECKPOINT]
                  [--class_balancing CLASS_BALANCING] [--image IMAGE]
                  [--continue_training CONTINUE_TRAINING] [--dataset DATASET]
                  [--load_data LOAD_DATA] [--act ACT]
                  [--crop_height CROP_HEIGHT] [--crop_width CROP_WIDTH]
                  [--batch_size BATCH_SIZE] [--num_val_images NUM_VAL_IMAGES]
                  [--h_flip H_FLIP] [--v_flip V_FLIP]
                  [--brightness BRIGHTNESS] [--rotation ROTATION]
                  [--model MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of epochs to train for
  --save SAVE           Interval for saving weights
  --gpu GPU             Choose GPU device to be used
  --mode MODE           Select "train", "test", or "predict" mode. Note that
                        for prediction mode you have to specify an image to
                        run the model on.
  --checkpoint CHECKPOINT
                        Checkpoint folder.
  --class_balancing CLASS_BALANCING
                        Whether to use median frequency class weights to
                        balance the classes in the loss
  --image IMAGE         The image you want to predict on. Only valid in
                        "predict" mode.
  --continue_training CONTINUE_TRAINING
                        Whether to continue training from a checkpoint
  --dataset DATASET     Dataset you are using.
  --load_data LOAD_DATA
                        Dataset loading type.
  --act ACT             True if sigmoid or false for softmax
  --crop_height CROP_HEIGHT
                        Height of cropped input image to network
  --crop_width CROP_WIDTH
                        Width of cropped input image to network
  --batch_size BATCH_SIZE
                        Number of images in each batch
  --num_val_images NUM_VAL_IMAGES
                        The number of images to used for validations
  --h_flip H_FLIP       Whether to randomly flip the image horizontally for
                        data augmentation
  --v_flip V_FLIP       Whether to randomly flip the image vertically for data
                        augmentation
  --brightness BRIGHTNESS
                        Whether to randomly change the image brightness for
                        data augmentation. Specifies the max bightness change.
  --rotation ROTATION   Whether to randomly rotate the image for data
                        augmentation. Specifies the max rotation angle.
  --model MODEL         The model you are using. Currently supports: FC-
                        DenseNet56, FC-DenseNet67, FC-DenseNet103, Encoder-
                        Decoder, Encoder-Decoder-Skip, RefineNet-Res50,
                        RefineNet-Res101, RefineNet-Res152, FRRN-A, FRRN-B,
                        MobileUNet, MobileUNet-Skip, PSPNet-Res50, PSPNet-
                        Res101, PSPNet-Res152, GCN-Res50, GCN-Res101, GCN-
                        Res152, DeepLabV3-Res50 DeepLabV3-Res101,
                        DeepLabV3-Res152, DeepLabV3_plus-Res50,
                        DeepLabV3_plus-Res101, DeepLabV3_plus-Res152, AdapNet,
                        custom, cesarNet, UNet

