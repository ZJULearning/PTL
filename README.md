# PTL: Progressive Transfer Learning for Person Re-identification
## Introduction

PTL is a model fine-tuning method for deep neural networks. 
It provides an efficient solution for the model fine-tuning task, and can improve the performance of the pre-trained model on the target dataset significantly.

This project is the implementation of the Batch-related Convolutional Cell (**BConv-Cell**) and the **MGN_PTL** network of our IJCAI-2019 paper - [Progressive Transfer Learning for Person Re-identification](TODO).

PTL method has been used by the CityBrain Group (Damo Academy, Alibaba Group) to help improve the model performance when using the pre-trained ReID models in a newly emerged scenario.

### BConv-Cell

* The key component of the MGN_PTL network is the BConv-Cell in bconv_cell.py
* The BConv-Cell can integrate with most deep neural networks to improve the model performance when using mini-batch training.
* This project only provides an example of its usage, feel free to explore.


## Performance

### Datasets
* [Market-1501](http://www.liangzheng.com.cn/Project/project_reid.html)
  
    Download using: 
        
      wget http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip <path/to/where/you/want>
      unzip <path/to/>/Market-1501-v15.09.15.zip
  
* [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)

  1. Download cuhk03 dataset from [here](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
  2. Unzip the file and you will get the cuhk03_release dir which include cuhk-03.mat
  3. Download "cuhk03_new_protocol_config_detected.mat" from [here](https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03) and put it with cuhk-03.mat. We need this new protocol to split the dataset.
  ```
  python utils/transform_cuhk03.py --src <path/to/cuhk03_release> --dst <path/to/save>
  ```
  NOTICE: You need to change num_classes in network depend on how many people in your train dataset! e.g. 751 in Market1501.

The data structure should look like:
    
  ```
  data/
      bounding_box_train/
      bounding_box_test/
      query/
  ```
        
### Compared person ReID methods

+ [DML](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf): *Deep Mutual Learning*
+ [HA-CNN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Harmonious_Attention_Network_CVPR_2018_paper.pdf): *Harmonious Attention Network for Person Re-identification*
+ [PCB](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yifan_Sun_Beyond_Part_Models_ECCV_2018_paper.pdf): *Beyond Part Models: Person Retrieval with Refined Part Pooling (and a Strong Convolutional Baseline)*
+ [PCB+RPP](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yifan_Sun_Beyond_Part_Models_ECCV_2018_paper.pdf): *Beyond Part Models: Person Retrieval with Refined Part Pooling (and a Strong Convolutional Baseline)*
+ [MGN](https://arxiv.org/pdf/1804.01438.pdf): *Learning Discriminative Features with Multiple Granularities for Person Re-Identification*

### Results

<table>
  <tr>
    <th>Method</th>
    <th colspan="2">Market-1501</th>
    <th colspan="2">DukeMTMC-reID</th>
    <th colspan="2">CUHK03(Detected)</th>
    <th colspan="2">CUHK03(Labelled)</th>
  </tr>
  <tr>
    <td></td>
    <td>mAP</td>
    <td>CMC-1</td>
    <td>mAP</td>
    <td>CMC-1</td>
    <td>mAP</td>
    <td>CMC-1</td>
    <td>mAP</td>
    <td>CMC-1</td>
  </tr>
  <tr>
    <td>DML</td>
    <td>70.51</td>
    <td>89.34</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>HA-CNN</td>
    <td>75.70</td>
    <td>91.20</td>
    <td>63.80</td>
    <td>80.50</td>
    <td>38.60</td>
    <td>41.70</td>
    <td>41.00</td>
    <td>44.40</td>
  </tr>
  <tr>
    <td>PCB</td>
    <td>77.30</td>
    <td>92.40</td>
    <td>65.30</td>
    <td>81.90</td>
    <td>54.20</td>
    <td>61.30</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>PCB+RPP</td>
    <td>81.60</td>
    <td>93.80</td>
    <td>69.20</td>
    <td>83.30</td>
    <td>57.50</td>
    <td>63.70</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>MGN</td>
    <td>86.90</td>
    <td>95.70</td>
    <td>78.40</td>
    <td>88.70</td>
    <td>66.00</td>
    <td>66.80</td>
    <td>67.40</td>
    <td>68.00</td>
  </tr>
  <tr>
    <td>MGN(reproduced)</td>
    <td>85.80</td>
    <td>94.60</td>
    <td>77.07</td>
    <td>87.70</td>
    <td>69.41</td>
    <td>71.64</td>
    <td>72.96</td>
    <td>74.07</td>
  </tr>
  <tr>
    <td><b>MGN_PTL</b></td>
    <td>87.34</td>
    <td>94.83</td>
    <td>79.16</td>
    <td>89.36</td>
    <td>74.22</td>
    <td>76.14</td>
    <td>77.31</td>
    <td>79.79</td>
  </tr>
</table>



NOTICE: The MGN(reproduced) is the reproduction of [MGN](https://arxiv.org/pdf/1804.01438.pdf). To our best knowledge, the official implementation of MGN has not released yet. Hence, the **MGN_PTL**
network used the MGN(reproduced) as backbone network. The code for MGN(reproduced) is in **mgn.py** 

## RUN
### Prerequisites

+ cudnn 7
+ CUDA 9
+ Pytorch v0.4.1
+ Python 2.7
+ torchvision
+ scipy
+ numpy
+ scikit_learn

### GPU usage

We used one Tesla P100 GPU in our experiments
* To run the MGN with batchid=4 and batchimage=4 cost 7819 MiB
* To run the MGN_PTL with batchid=4 and batchimage=4 cost 8819 MiB

### Weights
Pretrained weight download from **TODO** (Currently unavailable, the weight file will be released later)

### Train
You can specify more parameters in opt.py

* Train MGN_PTL
  ```
  CUDA_VISIBLE_DEVICES=0 python train_eval.py --arch mgn_ptl --mode train --usegpu --project_name 'temp_project' --data_path <path/to/Market-1501-v15.09.15> --lr 2e-4 --batchid 4 --epoch 450
  ```
* Train MGN
  ```
  CUDA_VISIBLE_DEVICES=0 python train_eval.py --arch mgn --mode train --usegpu --project_name 'temp_project' --data_path <path/to/Market-1501-v15.09.15> --lr 2e-4 --batchid 4 --epoch 450
  ```

### Evaluate
Use pretrained weight or your trained weight

* Evaluate MGN_PTL
  ```
  CUDA_VISIBLE_DEVICES=0 python train_eval.py --arch mgn_ptl --mode evaluate --usegpu --weight <path/to/weight/weight_name.pt> --data_path <path/to/Market-1501-v15.09.15>
   ```
* Evaluate MGN
  ```
  CUDA_VISIBLE_DEVICES=0 python train_eval.py --arch mgn --mode evaluate --usegpu --weight <path/to/weight/weight_name.pt> --data_path <path/to/Market-1501-v15.09.15>
   ```
   
## Reference

Reference to cite when you use PTL in a research paper:
**TODO**

## License
PTL is MIT-licensed.