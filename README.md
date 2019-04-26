# Gray Image Retrieval
This is the implementation of "Color Reference Recommendation" in the paper [**Deep Exemplar-based Colorization**](https://arxiv.org/abs/1807.06587).


## Introduction

Given a gray input image, **Gray Image Retrieval** is to search its similar images in semantic content and photometric luminance from ImageNet.

![image](https://github.com/hmmlillian/Gray-Image-Retrieval/blob/master/pipeline.jpg)

The proposed method consists of two ranking steps, **Global Ranking** which filters out dissimilar images within the same class, and **Local Ranking** which further prunes the candidates with large difference in spatial layouts and illuminance.

The input is either gray or color image (but it will be automatically converted to gray in the code), and the output is a text file saving the names of its top-N similar images in ImageNet.


## Getting Started

### Prerequisites
- Windows (64bit)
- NVIDIA GPU (CUDA 8.0 & CuDNN 6.0)
- Visual Studio 2013
- Python 2.7
- Matlab R2017
- OpenCV 2.4.10


### Build
(1) Compile pycaffe:
- Compile the pycaffe interface from [BVLC/caffe](https://github.com/BVLC/caffe/tree/windows);
- Put the compiled files under ```BVLC/caffe/tree/windows/python/``` to the folder ```build/pycaffe/```.
- We also provide our compiled files (https://drive.google.com/open?id=1DRFpb75eCCOynFtF2qbqVE16h-XFW27_) for testing.

(2) Install the Matlab engine for Python:
- Follow the instructions: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

(3) Compile Search.dll:
- Compile ```source/search/main.cpp``` (OpenCV 2.4.10 library required) in Visual Studio 2013 to generate Search.dll;
- Replace the precompiled Search.dll in ```build/search/```.


### Download Models
You need to download models and compressed feature database of ImageNet before running a demo.
- Go to ```model/vgg_19_gray_bn/``` folder and download:  
  https://www.dropbox.com/s/mnsxsfv5non3e81/vgg19_bn_gray_ft_iter_150000.caffemodel?dl=0
- Go to ```demo/ImageNet/``` folder and download: 
  https://drive.google.com/open?id=1FS04QjIowFN2iFWqP_5lfcG6GyGH5yJE


### Demo
- We prepare an example under the folder ```demo/``` with an input gray images folder ```test/imgs/``` and its output folder ```test/pairs/```.

- Go to root directory and run ```test.py```:
  ```
  python test.py --in_path [INPUT_IMAGE_FOLDER] --out_path [OUTPUT_FILE_FOLDER] --model_path [MODEL_FOLDER] --imagenet_path [IMAGENET_FEATURE_DATABASE] --gpu_id [GPU_ID]
  
  e.g., python test.py --in_path demo/test/imgs/ --out_path demo/test/pairs/ --model_path model/vgg_19_gray_bn/ --imagenet_path demo/ImageNet/ --gpu_id 0
  ```

### Note
To serve for the purpose of colorization reference recommendation, the input images will be converted to gray images for searching whether it is colorful or not. The classification on gray images may be worse than color images.


## Citation
If you find **Gray Image Retrieval** helpful for your research, please consider citing:
```
@article{he2018deep,
  title={Deep exemplar-based colorization},
  author={He, Mingming and Chen, Dongdong and Liao, Jing and Sander, Pedro V and Yuan, Lu},
  journal={ACM Transactions on Graphics (TOG)},
  volume={37},
  number={4},
  pages={47},
  year={2018},
  publisher={ACM}
}
```

