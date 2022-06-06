Turtle Recall: Conservation Challenge. 5th place solution
=========================================================

Competition: [link](https://zindi.africa/competitions/turtle-recall-conservation-challenge)  
Author: Igor Ivanov  
License: MIT  


Solution overview
=================

In order to ensure generalization ability I built my solution as an ensemble
of 6 models each of which was trained on a 5-fold stratified split.
For the same purpose I chose large deep architectures which have 
enough capacity to capture important features from the diverse dataset.
All models share the same multiclass classification formulation over 2265 classes 
with average pooling and softmax on top. Optimization was performed using 
categorical cross-entropy loss and Adam optimizer.
I used all available data for training i.e. joint set of training and extra images.
Raw model prediction contains 2265 probabilities. Any predicted `turtle_id` 
which does not belong to 100 original training individuals is considered a `new_turtle`.
Ensemble is computed as an arithmetic average of 30 predictions (6 models by 5 folds).

Architectures used:
- EfficientNet-v1-B7
- EfficientNet-v1-L2
- EfficientNet-v1-L2
- EfficientNet-v2-L
- EfficientNet-v2-XL
- BEiT-L

Architectures are implemented in the following repositories:
- https://github.com/qubvel/efficientnet
- https://github.com/leondgarse/keras_cv_attention_models

For augmentation I used rotations multiple of 45 degrees (with central crop) and flips.
For validation purposes I measured Accuracy and MAP5 over 2265 classes.
Software stack is based on Tensorflow and Keras.
All hyperparameters are listed in a dedicated section on the top 
of the `run.py` file and can be passed as command line arguments.  


Results
=======

Each score in the table is an average of 5 folds.  
Suffix `2265` means that metric uses 2265 unique turtle ids (100 training + extra)  
Suffix `101` means that metric uses 101 unique turtle ids (100 training + 1 `new_turtle`)  

| Model                    | CV-acc1-2265 | CV-map5-2265 | Public-LB-map5-101 | Private-LB-map5-101  |
|--------------------------|--------------|--------------|--------------------|----------------------|
| run-20220310-1926-ef1b7  | 0.8731       | 0.9067       | 0.9523             | 0.9567               |
| run-20220316-1310-beitl  | 0.8896       | 0.9202       | 0.9611             | 0.9317               |
| run-20220317-1954-ef1l2  | 0.8782       | 0.9112       | 0.9543             | 0.9501               |
| run-20220318-1121-ef2xl  | 0.8553       | 0.8928       | 0.9421             | 0.9332               |
| run-20220322-2024-ef1l2  | 0.8720       | 0.9056       | 0.9625             | 0.9514               |
| run-20220325-1527-ef2l   | 0.8829       | 0.9151       | 0.9557             | 0.9545               |
| -                        |              |              |                    |                      |
| Ensemble                 | 0.9320       | 0.9503       | 0.9875             | 0.9648               |



Conclusions:
============

1) Solution generalizes well between public and private test sets
   despite very small test data size (147 and 343 examples respectively).
   As a result I was able to retain high position in both leaderboards:
   2nd place public, 5th place private.

2) Ensembling gives stable significant improvement (about 0.01-0.03) 
   observed by all metrics on all subsets of data (public/private).

3) Combination of GeM pooling and ArcFace loss is a popular approach in the tasks dealing with image similarity.
   But in this task I did not see an improvement from this approach in my experiments.


Hardware
========

Training: TPUv3-8, 4 CPU, 16 GB RAM, 500 GB HDD  
Training time: 100 hours total  

Inference: V100-16GB GPU, 4 CPU, 16 GB RAM, 500 GB HDD  
Inference time: 30 minutes total  


Software
========

- Ubuntu 18.04  
- Python: 3.9.7
- CUDA: 11.2
- cuDNN: 8.1.1
- Tensorflow: 2.8.0


Demo
====

The following example `solution/notebook/notebook.ipynb` demonstrates
how to download pretrained weights and infer any single image.
Leaderboard score of this model is 0.9214 (EffNet-B7, single fold).


Steps to reproduce
==================

```
# Install

cd $HOME
git clone https://github.com/vecxoz/turtle-recall
mv turtle-recall solution
conda create -y --name py397 python=3.9.7
conda activate py397
pip install tensorflow==2.8.0 tensorflow-addons numpy pandas \
scikit-learn h5py efficientnet keras-cv-attention-models cloud-tpu-client

# Prepare data

mkdir -p $HOME/solution/data
cd $HOME/solution/data

curl -L -O https://storage.googleapis.com/dm-turtle-recall/train.csv
curl -L -O https://storage.googleapis.com/dm-turtle-recall/extra_images.csv
curl -L -O https://storage.googleapis.com/dm-turtle-recall/test.csv
curl -L -O https://storage.googleapis.com/dm-turtle-recall/sample_submission.csv
curl -L -O https://storage.googleapis.com/dm-turtle-recall/images.tar

mkdir images
tar xf images.tar -C images
rm images.tar
cd $HOME/solution
python3 create_tfrecords.py --data_dir=$HOME/solution/data --out_dir=$HOME/solution/data/tfrec

# Training

# Please remove all weights from previous runs if present.
# All hyperparameters are configured for training on TPUv3-8.
# To train on GPU (or several GPUs) set the following arguments in `run_training.sh`:
# --tpu_ip_or_name=None
# --data_tfrec_dir=$HOME/solution/data/tfrec
# and adjust batch size and learning rate accordingly.
# To use mixed precision set:
# --mixed_precision=mixed_float16
# Note. If the target system runs out of RAM you can disable data caching.
# Use argument `cache=False` in calls to `init_tfdata` function in each `run.py` file.

bash run_training.sh

# Inference

bash run_inference.sh
# Submission will appear as $HOME/solution/submission.csv
```


Acknowledgement
===============

Thanks to [TRC program](https://sites.research.google/trc/about/) 
I had an opportunity to run experiments on TPUv3-8.

