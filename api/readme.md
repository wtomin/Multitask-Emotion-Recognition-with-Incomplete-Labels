# Multitask Emotion Recognition api
---
This repository contain all scripts that needed for running Multi-task Emotion Recognition api for video files. Using this this api, you are able to get three types of emotion predictions, including 8 FAUs(Facial Action Units), 7 basic facial expressions and VA (Valence and Arousal) of each frame of facial images in the input video.

# Get Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Prerequisites
My recommendation is to use Anaconda to create an virtual environment. For example:
```
conda create --name myenv python=3.6
```

And then activate the virtual environment by:
```
conda activate myenv
```

- CUDA
My recommendation is to used Anaconda to install cudatoolkit 10.1, torch and torchvision:
```
conda install pytorch torchvision cudatoolkit=10.1
```

- OpenFace

If you have already installed OpenFace, you can skip the installation below. You only need to replace pass the executable file path `FeatureExtraction` to `EmotionAPI()`.

To install OpenFace in the root directory of this project:
```
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
```
Then download the needed models by:
```
bash ./download_models.sh
```
It's better to install some dependencies required by OpenCV before you run 'install.sh':
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake pkg-config # Install developer tools used to compile OpenCV
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev #  Install libraries and packages used to read various image formats from disk
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev # Install a few libraries used to read video formats from disk
```
And then install OpenFace by:
```
sudo bash ./install.sh
```
If this shell script fails to install it correctly, you can follow the instruction [here](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation#advanced-ubuntu-installation-if-not-using-installsh-or-if-it-fails).

- pytorch-benchmarks
Download pytorch-benchmarks in the root directory of your project:
```
git clone https://github.com/albanie/pytorch-benchmarks.git
```
Download the three folders in this [link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/ddeng_connect_ust_hk/ElE4ifQSxLBCgtfQLvgE5Z8B8wGo-SpoUJc-So_3FruMcg?e=LL3u6V) and save the three folders under `pytorch-benchmarks/models`.

- Other requirements

Install other requirements by:
```
pip install -r requirements.txt
```

- Download pretrained models

Download pretrained Multitask CNN models and Multitask CNN-RNN models by:

```
bash download_pretrained_weights_aff_wild2.sh
```

# How to use
There is an example script `run_example.py`.

Run 
```
python run_example.py
```
The prediction results will be saved to `predictions.csv`. 

Example:
```
   frames_ids  AU1  AU2  AU4  AU6  ...  AU20  AU25  EXPR   valence   arousal
0           1    0    0    1    1  ...     0     0     4  0.499485  0.522120
1           2    0    0    1    1  ...     0     0     4  0.488518  0.504697
2           3    0    0    1    1  ...     0     0     4  0.445970  0.469693
3           4    0    0    1    1  ...     0     0     4  0.421242  0.448395
4           5    0    0    1    1  ...     0     0     4  0.430781  0.450103

[5 rows x 12 columns]

``` 

# TODO
- To generate a video will where three types of emotion predictions are drawn.
