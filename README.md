# Co-attention CNNs for Unsupervised Object Co-segmentation

## Description

This is my partial implementation of the paper: **Co-attention CNNs for Unsupervised Object Co-segmentation** by Kuang-Jui Hsu, Yen-Yu Lin, and Yung-Yu Chuang (https://www.csie.ntu.edu.tw/~cyy/publications/papers/Hsu2018CAC.pdf)

The system consists of two networks:
- generator (FCN32)
- feature extractor (ResNet50)

The generator produces masks, that are used to generate objects and background images, by simple element-wise product:
- object_image = image * mask
- background_image = image * (1 - mask)

 Next, those images are provided to the feature extractor to generate features that are used in the loss function. Only the weights of the generator are adjusted during the training. The so-called co-segmentation loss aims to decrease the Euclidian distance between features of segmented objects and increase the Euclidian distance between features of objects and backgrounds. This loss makes the generator to produce masks that cover only common objects in the images.

I implemented only one part of loss function - co-segmentation loss, however, it provides decent results.  For better results, an additional part of loss function - mask loss - is needed, which could be interpreted as a regularization term. The method is very sensitive to hyperparameter choice. 


## Results
![](img/example0.png)
![](img/example1.png)
![](img/example2.png)
![](img/example4.png)

## Dependencies
- Pytorch 1.3.1
- Tensorboard
- Numpy
- Matplotlib
## Details
I used FCN32s implementation available at https://github.com/pochih/FCN-pytorch

The system was trained on the Internet Dataset http://people.csail.mit.edu/mrub/ObjectDiscovery/  
I used the subset of this dataset, that is stored in the following folders:
- Airplane100
- Horse100
- Car100


To train the model create the simple folder structure:  
```
.
+-- data
|    |--Airplane100
|    |--Horse100
|    |--Car100
.
```
