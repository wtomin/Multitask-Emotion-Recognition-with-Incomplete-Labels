# Multitask-Emotion-Recognition-with-Incomplete-Labels
This is the repository containing the solution for FG-2020 ABAW Competition

Pretrained models can be downloaded through this [link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/ddeng_connect_ust_hk/EnX91m9VSHlFobaIag82W_8B3YRkir97H1QmiUlkZu1zAw?e=LGgDNE)

 [Paper](https://www.computer.org/csdl/pds/api/csdl/proceedings/download-article/1kecJ0EYZxK/pdf),   [Presentation](https://hkustconnect-my.sharepoint.com/:v:/g/personal/ddeng_connect_ust_hk/ETn3jr7KVX1JjJ_MP8Ua8MEBIEP2WcJyyviYApd951qh4g?e=KOkr8Z)
 
We aim for a unifed model to solve three tasks: Facial Action Units (FAU) prediction, Facial Expression (7 basic emotions) prediction, Valence and Arousal prediction. For abbreviation, we refer to them as FAU, EXPR and VA.

*UPDATES*: The challenge [leaderboard](https://ibug.doc.ic.ac.uk/resources/fg-2020-competition-affective-behavior-analysis/) has been released. Our solution won two challenege tracks (FAU and VA) among six teams!

---
## DEMO: We made our latest demo available!
[![Watch the video](https://github.com/wtomin/Multitask-Emotion-Recognition-with-Incomplete-Labels/blob/master/imgs/thumnail.jpg)](https://youtu.be/0-dnW0Rb5_U)

To make such a demo, modify the `video_file` in `emotion_demo.py` and then run `python emotion_demo.py`. The output video will be saved under the `save_dir`.

To run this demo, [MTCNN](https://github.com/ipazc/mtcnn) must be installed.

---
## Data Balancing

Before training, we change the data distribution of experiment datasets by (1) importing external datasets, such as the [DISFA](http://mohammadmahoor.com/disfa/) dataset for FAU, the [ExpW](http://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html) dataset for EXPR, and the [AFEW-VA](https://ibug.doc.ic.ac.uk/resources/afew-va-database/) dataset for VA; (2) resampling the minority class and the majority class. Our purpose is to create a more balanced data distribution for each individual class.

This the data disribution of the Aff-wild2 dataset, the DISFA dataset and the merged dataset. We resampled the merged dataset using ML-ROS, which is short for [Multilabel Randomly Oversampling](https://www.sciencedirect.com/science/article/pii/S0925231215004269)

<img src="https://github.com/wtomin/A-Multitask-Solution-for-FAU-EXPR-VA/blob/master/imgs/AU_distribution.png" width="500">

This the data distribution of the Aff-wild2 dataset, the ExpW dataset and the merged dataset. We resample the merged dataset to ensure the instances of each class have the same probability of appearing in one epoch.

<img src="https://github.com/wtomin/A-Multitask-Solution-for-FAU-EXPR-VA/blob/master/imgs/EXPR_distribution.png" width="500">

This the data distribution of the Aff-wild2 dataset, the AFEW-VA dataset and the merged dataset. We discretize the continuous valence/arousal scores in [-1, 1] into 20 bins of the same width. We treat each bin as a category, and apply the oversampling/undersampling strategy.

<img src="https://github.com/wtomin/A-Multitask-Solution-for-FAU-EXPR-VA/blob/master/imgs/VA_distribution.png" width="500">

## Learning With Partial Labels

For the current datasets, each dataset only contain one type of labels (FAU, EXPR or VA). Therefore we propose an algorithm for a deep neural network to learn multitask from partial labels.
The algorithm has two steps: firstly, we train a teacher model to perform all three tasks, where each instance is trained by the ground truth label of its corresponding task. Secondly, we refer to the outputs of the teacher model as the soft labels. We use the soft labels and the ground truths to train the student model. 

This is the diagram for our proposed algorithm. Given the input images of three tasks <img src="https://latex.codecogs.com/gif.latex?X&space;=&space;\{&space;X^{(1)},&space;X^{(2)},&space;X^{(3)}\}" title="X = \{ X^{(1)}, X^{(2)}, X^{(3)}\}" /> and the ground truths of three tasks <img src="https://latex.codecogs.com/gif.latex?y&space;=&space;\{&space;y^{(1)},&space;y^{(2)},&space;y^{(3)}\}" title="y = \{ y^{(1)}, y^{(2)}, y^{(3)}\}" />, we first train the teacher model using the teacher loss between the teacher outputs <img src="https://latex.codecogs.com/gif.latex?\hat{y_t}^{(i)}" title="\hat{y_t}^{(i)}" /> and the ground truth <img src="https://latex.codecogs.com/gif.latex?y^{(i)}" title="y^{(i)}" />. Secondly, we train the student model using the student loss which consists of two parts: one is calcaluted from the teacher outputs <img src="https://latex.codecogs.com/gif.latex?\hat{y_t}^{(i)}" title="\hat{y_t}^{(i)}" /> and the student outputs <img src="https://latex.codecogs.com/gif.latex?\hat{y_s}^{(i)}" title="\hat{y_s}^{(i)}" />, another is calculated from the ground truth <img src="https://latex.codecogs.com/gif.latex?y^{(i)}" title="y^{(i)}" /> and the student outputs <img src="https://latex.codecogs.com/gif.latex?\hat{y_s}^{(i)}" title="\hat{y_s}^{(i)}" />.

<img src="https://github.com/wtomin/A-Multitask-Solution-for-FAU-EXPR-VA/blob/master/imgs/algorithm.png" width="700">

---
## Requiremnets
* Pytorch 1.3.1 or higher version
* Numpy
* [pytorch benchmark](https://github.com/albanie/pytorch-benchmarks)
* pandas, pickle, matplotlib 

## How to replicate our results
1. Download all required datasets, crop and align face images;

2. Create the annotation files for each dataset, using the script in `create_annotation_file` directory;

3. Change the annotation file paths in the `Multitask-CNN(Multitask-CNN-RNN)/PATH/__init__.py`;

4. Training: For Multitask-CNN, run `python train.py --force_balance --name image_size_112_n_students_5 --image_size 112 --pretrained_teacher_model path-to-teacher-model-if-exists`, the argument `name` is experiment name (save path), the `--force_balance` will make the sampled dataset more balanced.  
For Multitask-CNN-RNN, run `python train.py --name image_size_112_n_students_5_seq_len=32 --image_size 112 --seq_len 32 --frozen --pretrained_resnet50_model path-to-the-pretrained-Multitask-CNN-model --pretrained_teacher_model path-to-teacher-model-if-exists `

5. Validation: Run the `python val.py --name image_size_112_n_students_5 --image_size 112 --teacher_model_path path-to-teacher-model --mode Validation --ensemble ` for Multitask-CNN, and run `python val.py --name image_size_112_n_students_5_seq_len=32 --image_size 112 --teacher_model_path path-to-teacher-model --pretrained_resnet50_model path-to-the-pretrained-Multitask-CNN-model  --mode Validation --ensemble --seq_len 32` for Multitask-CNN-RNN.

6. From the results on the validation set, we obtain the best AU thresholds on the validation set.  
Modify this line `best_thresholds_over_models = [] ` in the `test.py` to the best thresholds on the validation set.

7. Testing: run `python test.py --name image_size_112_n_students_5 --image_size 112 --teacher_model_path path-to-teacher-model --mode Test --save_dir Predictions --ensemble` for Multitask-CNN, and run `python test.py --name image_size_112_n_students_5_seq_len=32 --image_size 112 --teacher_model_path path-to-teacher-model --pretrained_resnet50_model path-to-the-pretrained-Multitask-CNN-model  --mode Test --ensemble --seq_len 32` for Multitask-CNN-RNN.

---
## How to implement our model on your data
1. Download the pretrained CNNs and unzip them.

2. Crop and align face images, save them to a directory.

3. For CNN model: `python run_pretrained_model.py --image_dir directory-containing-sequence-of-face-images --model_type CNN --batch_size 12 --eval_with_teacher --eval_with_students  --save_dir save-directory --workers 8 --ensemble`. For CNN-RNN model: `python run_pretrained_model.py --image_dir directory-containing-sequence-of-face-images --model_type CNN-RNN --seq_len 32 --batch_size 6 --eval_with_teacher --eval_with_students  --save_dir save-directory --workers 8 --ensemble`

---
## If you are interested in our work, please cite
```
@inproceedings{deng2020multitask,
  title={Multitask Emotion Recognition with Incomplete Labels},
  author={Deng, Didan and Chen, Zhaokang and Shi, Bertram E},
  booktitle={2020 15th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2020)(FG)},
  pages={828--835},
  organization={IEEE Computer Society}
}
```
