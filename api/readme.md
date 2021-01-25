# Multitask Emotion Recognition api
---
This repository contain all scripts that needed for running Multi-task Emotion Recognition api for video files. Using this this api, you are able to get three types of emotion predictions, including 8 FAUs(Facial Action Units), 7 basic facial expressions and VA (Valence and Arousal) of each frame of facial images in the input video.

# Get Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Prerequisites
My recommendation is to use Anaconda to create an virtual environment. For example:
```
conda create --name myenv --python=3.6
```

And then activate the virtual environment by:
```
conda activate myenv
```

- CUDA
My recommendation is to used Anaconda to install cudatoolkit 10.1:
```
conda install cudatoolkit=10.1 -c pytorch 
```

- OpenFace

To install OpenFace in the root directory of this project:
```
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
```
Then download the needed models by:
```
bash download_models.sh
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
bash install.sh
```

- pytorch-benchmarks
Install pytorch-benchmarks in the root directory of your project:
```
git clone https://github.com/albanie/pytorch-benchmarks.git
```
Create a directory in pytorch-benchmarks to store the resnet50 model and weights:
```
mkdir pytorch-benchmarks/ferplus/
```
Then download the resnet50 model and weights by
```
wget -P pytorch-benchmarks/ferplus/ http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/resnet50_ferplus_dag.py

wget -P pytorch-benchmarks/ferplus/ http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/resnet50_ferplus_dag.pth 
```

- Other requirements

Install other requirements by:
```
pip install -r requirements.txt
```

- Download pretrained models

Download pretrained Multitask CNN models and Multitask CNN-RNN models by:

```
bash download_models.sh
```

# How to use
There is an example script `run_example.py`.

Run 
```
python run_example.py
```
The prediction results will be saved to `predictions.csv`.  

In addition, if you pass `video=True` to `API`, a video will be generated where three types of emotion predictions are drawn as plots.
