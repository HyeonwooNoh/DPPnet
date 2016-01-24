## DPPnet: Image Question Answering using Convolutional Neural Network with Dynamic Parameter Prediction

Created by [Hyeonwoo Noh](http://cvlab.postech.ac.kr/~hyeonwoonoh/), [Paul Hongsuck Seo](https://sites.google.com/site/paulhseo/) and [Bohyung Han](http://cvlab.postech.ac.kr/~bhhan/) at [POSTECH cvlab](http://cvlab.postech.ac.kr/lab/)

Project page: [http://cvlab.postech.ac.kr/research/dppnet/]

### Introduction

DPPnet is state-of-the-art Image Question Answering algorithm using dynamic parameter prediction to handle various types of questions.

Detailed description of the system will be provided by our technical report [arXiv tech report](http://arxiv.org/abs/1511.05756)

### Citation

If you're using this code in a publication, please cite our papers.

    @article{noh2015image,
      title={Image Question Answering using Convolutional Neural Network with Dynamic Parameter Prediction},
      author={Noh, Hyeonwoo and Seo, Paul Hongsuck and Han, Bohyung},
      journal={arXiv preprint arXiv:1511.05756},
      year={2015}
    }

### Licence

This software is for research purpose only.
Check LICENSE file for details.

### System Requirements

  * This software is tested on Ubuntu 14.04 LTS (64bit).
  * At least 12GB gpu memory is required (NVIDIA tital-x gpu is used for training).

### Dependencies

  * torch [https://github.com/torch/distro]
  * loadcaffe [https://github.com/szagoruyko/loadcaffe]
  * xxhash [install: luarocks install xxhash]

### Setup

Run "setup.sh" for setting up.

### Testing

Scripts for testing is in "006\_test\_DPPnet". Use following commands for testing.
  0. Run ./gen\_simulinks.sh
  0. Run th vqa\_test.lua
  0. Results will be saved in "006\_test\_DPPnet/save\_result\_vqa\_test/results/"

### Training

Following steps are required for training.
  0. Train DPPnet with fixed cnn feature (004\_train\_DPPnet\_fixed\_cnn)
  0. Finetune CNN from the model trained in the previous step (005\_train\_DPPnet\_finetune\_cnn)

### Directories
  * 001\_porting\_VQA\_data: porting VQA data for torch implementation
  * 002\_extract\_image\_features: extracting VGG16 features from MSCOCO
  * 003\_skipthoughts\_porting: porting model parameters for [skipthoughts](https://github.com/ryankiros/skip-thoughts)
  * 004\_train\_DPPnet\_fixed\_cnn: training DPPnet with extracted feature
  * 005\_train\_DPPnet\_finetune\_cnn: fine-tuning cnn from model trained 004
  * cache: We cache loaded vqa questions ans vocabularies to reduce the time for loading
  * data: data used for training / testing
  * model: trained model parameters, model definitions, layer implementations
  * utils: utilities (loading training data, loading models ...)



