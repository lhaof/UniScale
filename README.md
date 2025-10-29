# Universal Scale Transformer for Histology Image Segmentation

## 📖 Introduction

Histology image segmentation is a critical prerequisite to pathological diagnosis. Accurate segmentation of these images can significantly aid physicians by facilitating quicker and more precise diagnostic decisions. A notable challenge in this area arises from the fact that different objects within histology images require segmentation at varying magnifications. However, most existing models are typically limited to performing segmentation at one pre-determined magnification. In this paper, we propose a novel universal scale transformer model (UniScaleFormer), which employs a scale-aware approach and integrates textual input to uniformly segment objects in histology images across various magnifications. Our method adopts an end-to-end architecture and utilizes a candidate mask query mechanism, specifically designed to identify and distinguish objects at different image scales. Moreover, we develop a Scale-Aware Module that enhances our network's ability to recognize the magnification level of input histology images, by using a scale query with extracted visual features. Then the scale query is integrated with mask queries, facilitating the incorporation of scale information. Experimental results demonstrate that the proposed method effectively achieves competitive results on various segmentation benchmarks at different magnifications. 

## 🔧 Dependencies and Installation
```shell
git clone https://github.com/lhaof/UniCell.git
cd mmdetection
conda create -n UniScaleFormer python=3.10
conda activate UniScaleFormer
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -e .
```

## 📖 Dataset Preparation


## 🚀 Training
**step 0.** Modify the `root_p` in `projects/MultiScaleMask2Former/configs/KFOLD_UniScaleFormer/SwinB_FOLD_0.py` and `YOUR_PATH` in `train.sh`, Then
```shell
bash train.sh
```

## ✍️ Inference
**step 0.** Download the UniScaleFormer model from [google drive].

**step 1.** Modify the config_path and checkpoint_path then run
```
bash tools/dist_test.sh config_path checkpoint_path 8
```