# Road_extraction
Attention Unet and Deep Unet implementation for road extraction using multi-gpu model tensorflow

Several variations of Deep U-Net were tested with extra layers and extra convolutions. Nevertheless, the model that outperformed all of them was the Attention U-Net: Learning Where to Look for the Pancreas. I have added an extra tweak improving even further performance, switching the convolution blocks to the residual blocks
https://arxiv.org/abs/1804.03999


![Attention U-Net](https://github.com/LeeJunHyun/Image_Segmentation/blob/master/img/AttU-Net.png)

