# RCIL
**[CVPR2022] Representation Compensation Networks for Continual Semantic Segmentation**<br>
Chang-Bin Zhang<sup>1</sup>, Jia-Wen Xiao<sup>1</sup>, Xialei Liu<sup>1</sup>, Ying-Cong Chen<sup>2</sup>, Ming-Ming Cheng<sup>1</sup><br>
<sup>1</sup> <sub>College of Computer Science, Nankai University</sub><br />
<sup>2</sup> <sub>The Hong Kong University of Science and Technology</sub><br />

[![Conference](https://img.shields.io/badge/CVPR-2022-blue)]()
[![Paper](https://img.shields.io/badge/arXiv-2203.05402-brightgreen)](https://arxiv.org/abs/2203.05402)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/representation-compensation-networks-for/overlapped-100-50-on-ade20k)](https://paperswithcode.com/sota/overlapped-100-50-on-ade20k?p=representation-compensation-networks-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/representation-compensation-networks-for/overlapped-100-5-on-ade20k)](https://paperswithcode.com/sota/overlapped-100-5-on-ade20k?p=representation-compensation-networks-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/representation-compensation-networks-for/overlapped-50-50-on-ade20k)](https://paperswithcode.com/sota/overlapped-50-50-on-ade20k?p=representation-compensation-networks-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/representation-compensation-networks-for/overlapped-100-10-on-ade20k)](https://paperswithcode.com/sota/overlapped-100-10-on-ade20k?p=representation-compensation-networks-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/representation-compensation-networks-for/domain-1-1-on-cityscapes)](https://paperswithcode.com/sota/domain-1-1-on-cityscapes?p=representation-compensation-networks-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/representation-compensation-networks-for/domain-11-1-on-cityscapes)](https://paperswithcode.com/sota/domain-11-1-on-cityscapes?p=representation-compensation-networks-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/representation-compensation-networks-for/domain-11-5-on-cityscapes)](https://paperswithcode.com/sota/domain-11-5-on-cityscapes?p=representation-compensation-networks-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/representation-compensation-networks-for/overlapped-15-1-on-pascal-voc-2012)](https://paperswithcode.com/sota/overlapped-15-1-on-pascal-voc-2012?p=representation-compensation-networks-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/representation-compensation-networks-for/disjoint-15-1-on-pascal-voc-2012)](https://paperswithcode.com/sota/disjoint-15-1-on-pascal-voc-2012?p=representation-compensation-networks-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/representation-compensation-networks-for/disjoint-10-1-on-pascal-voc-2012)](https://paperswithcode.com/sota/disjoint-10-1-on-pascal-voc-2012?p=representation-compensation-networks-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/representation-compensation-networks-for/disjoint-15-5-on-pascal-voc-2012)](https://paperswithcode.com/sota/disjoint-15-5-on-pascal-voc-2012?p=representation-compensation-networks-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/representation-compensation-networks-for/overlapped-10-1-on-pascal-voc-2012)](https://paperswithcode.com/sota/overlapped-10-1-on-pascal-voc-2012?p=representation-compensation-networks-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/representation-compensation-networks-for/overlapped-15-5-on-pascal-voc-2012)](https://paperswithcode.com/sota/overlapped-15-5-on-pascal-voc-2012?p=representation-compensation-networks-for)

## Method
<img width="1230" alt="截屏2022-04-09 上午1 02 44" src="https://user-images.githubusercontent.com/35215543/162488465-73c56e73-8d5b-4406-941f-85497673c419.png">


## Update
- init code for Classification
- ~~We have fixed bugs in the repository~~
- ~~add training scripts for ADE20K~~
- ~~09/04/2022 init code for segmentation~~
- ~~09/04/2022 init readme~~



## Benchmark and Setting
There are two commonly used settings, ```disjoint``` and ```overlapped```.
In the ```disjoint``` setting, assuming we know all classes in the future, the images in the current training step do not contain any classes in the future. The ```overlapped``` setting allows potential classes in the future to appear in the current training images. We call each training on the newly added dataset as a step. Formally, ```X-Y``` denotes the continual setting in our experiments, where ```X``` denotes the number of classes that we need to train in the first step. In each subsequent learning step, the newly added dataset contains ```Y``` classes. 

There are some settings reported in our paper. You can also try it on other any custom settings.
- Continual Class Segmentation:
    1. PASCAL VOC 2012 dataset:
        - 15-5 overlapped
        - 15-5 disjoint
        - 15-1 overlapped
        - 15-1 disjoint
        - 10-1 overlapped
        - 10-1 disjoint
    2. ADE20K dataset:
        - 100-50 overlapped
        - 100-10 overlapped
        - 50-50 overlapped
        - 100-5 overlapped
- Continual Domain Segmentation:
    1. Cityscapes:
        - 11-5
        - 11-1
        - 1-1

- Extension Experiments on Continual Classification
    1. ImageNet-100
        - 50-10

## Performance 
- Continual Class Segmentation on PASCAL VOC 2012

| Method | Pub.       | 15-5 disjoint | 15-5 overlapped | 15-1 disjoint | 15-1 overlapped | 10-1 disjoint | 10-1 overlapped |
| ------ | ---------- | ------------- | --------------- | ------------- | --------------- | ------------- | --------------- |
| LWF    | TPAMI 2017 | 54.9          | 55.0            | 5.3           | 5.5             | 4.3           | 4.8             |
| ILT    | ICCVW 2019 | 58.9          | 61.3            | 7.9           | 9.2             | 5.4           | 5.5             |
| MiB    | CVPR 2020  | 65.9          | 70.0            | 39.9          | 32.2            | 6.9           | 20.1            |
| SDR    | CVPR 2021  | 67.3          | 70.1            | 48.7          | 39.5            | 14.3          | 25.1            |
| PLOP   | CVPR 2021  | 64.3          | 70.1            | 46.5          | 54.6            | 8.4           | 30.5            |
| Ours   | CVPR 2022  | 67.3          | 72.4            | 54.7          | 59.4            | 18.2          | 34.3            |


- Continual Class Segmentation on ADE20K

| Method | Pub.       | 100-50 overlapped | 100-10 overlapped | 50-50 overlapped | 100-5 overlapped |
| ------ | ---------- | ----------------- | ----------------- | ---------------- | ---------------- |
| ILT    | ICCVW 2019 | 17.0              | 1.1               | 9.7              | 0.5              |
| MiB    | CVPR 2020  | 32.8              | 29.2              | 29.3             | 25.9             |
| PLOP   | CVPR 2021  | 32.9              | 31.6              | 30.4             | 28.7             |
| Ours   | CVPR 2022  | 34.5              | 32.1              | 32.5             | 29.6             |


- Continual Domain Segmentation on Cityscapes

| Method | Pub.       | 11-5 | 11-1 | 1-1  |
| ------ | ---------- | ---- | ---- | ---- |
| LWF    | TPAMI 2017 | 59.7 | 57.3 | 33.0 |
| LWF-MC | CVPR 2017  | 58.7 | 57.0 | 31.4 |
| ILT    | ICCVW 2019 | 59.1 | 57.8 | 30.1 |
| MiB    | CVPR 2020  | 61.5 | 60.0 | 42.2 |
| PLOP   | CVPR 2021  | 63.5 | 62.1 | 45.2 |
| Ours   | CVPR 2022  | 64.3 | 63.0 | 48.9 |



## Dataset Prepare
- PASCVAL VOC 2012  
    ```sh data/download_voc.sh```
- ADE20K  
    ```sh data/download_ade.sh```
- Cityscapes  
    ```sh data/download_cityscapes.sh```


## Environment
1. ```conda install --yes --file requirements.txt```
2. Install [inplace-abn](https://github.com/mapillary/inplace_abn)



## Training
1. Dowload pretrained model from [ResNet-101_iabn](https://github.com/arthurdouillard/CVPR2021_PLOP/releases/download/v1.0/resnet101_iabn_sync.pth.tar) to ```pretrained/```
2. We have prepared some training scripts in ```scripts/```. You can train the model by
```
sh scripts/voc/rcil_10-1-overlap.sh
```

## Inference
You can simply modify the bash file by adding ```--test```, like
```
CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} run.py --data xxx ... --test
```






## Reference
If this work is useful for you, please cite us by:
```
@inproceedings{zhang2022representation,
  title={Representation Compensation Networks for Continual Semantic Segmentation},
  author={Zhang, Chang-Bin and Xiao, Jia-Wen and Liu, Xialei and Chen, Ying-Cong and Cheng, Ming-Ming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7053--7064},
  year={2022}
}
```

## Connect
If you have any questions about this work, please feel easy to connect with us (zhangchbin ^ mail.nankai.edu.cn or zhangchbin ^ gmail.com).



## Thanks
This code is heavily borrowed from [[MiB]](https://github.com/fcdl94/MiB) and [[PLOP]](https://github.com/arthurdouillard/CVPR2021_PLOP).




## Awesome Continual Segmentation
There is a collection of AWESOME things about continual semantic segmentation, including papers, code, demos, etc. Feel free to pull request and star.

### 2022
- Representation Compensation Networks for Continual Semantic Segmentation [[CVPR 2022]](https://arxiv.org/abs/2203.05402) [[PyTorch]](https://github.com/zhangchbin/RCIL)
- Self-training for Class-incremental Semantic Segmentation [[TNNLS 2022]](https://arxiv.org/abs/2012.03362) [PyTorch]
- Uncertainty-aware Contrastive Distillation for Incremental Semantic Segmentation [[TPAMI 2022]](https://arxiv.org/pdf/2203.14098.pdf) [[PyTorch]]


### 2021
- PLOP: Learning without Forgetting for Continual Semantic Segmentation [[CVPR 2021]](https://arxiv.org/abs/2011.11390) [[PyTorch]](https://github.com/arthurdouillard/CVPR2021_PLOP)
- Continual Semantic Segmentation via Repulsion-Attraction of Sparse and Disentangled Latent Representations [[CVPR2021]](https://arxiv.org/abs/2103.06342) [[PyTorch]](https://github.com/LTTM/SDR)
- An EM Framework for Online Incremental Learning of Semantic Segmentation [[ACM MM 2021]](https://arxiv.org/pdf/2108.03613.pdf) [[PyTorch]](https://github.com/Rhyssiyan/Online.Inc.Seg-Pytorch)
- SSUL: Semantic Segmentation with Unknown Label for Exemplar-based Class-Incremental Learning [[NeurIPS 2021]](https://proceedings.neurips.cc/paper/2021/file/5a9542c773018268fc6271f7afeea969-Paper.pdf) [[PyTorch]](https://github.com/clovaai/SSUL)


### 2020
- Modeling the Background for Incremental Learning in Semantic Segmentation [[CVPR 2020]](https://arxiv.org/abs/2002.00718) [[PyTorch]](https://github.com/fcdl94/MiB)

### 2019
- Incremental Learning Techniques for Semantic Segmentation [[ICCV Workshop 2019]](https://arxiv.org/abs/1907.13372) [[PyTorch]](https://github.com/LTTM/IL-SemSegm)





