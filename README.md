# Hyper-Modality Enhancement for Multimodal Sentiment Analysis with Missing Modalities
We propose the HME to address the modality missingness in real-world scenarios, which is accepted by [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/116212).


### The Framework of HME:
![image](https://github.com/YetZzzzzz/HME/blob/main/framework.png)
Figure 1: Visualization of HME framework. It consists of three modules: Pre-Processing Module, Hyper-Modality Representation Generation Module and Multimodal Hyper-Modality Fusion Module.



### Datasets:
**Please move the following datasets into directory ./datasets/.**

The CMU-MOSI and CMU-MOSEI datasets can be downloaded according to [MIB](https://github.com/TmacMai/Multimodal-Information-Bottleneck) and [MAG](https://github.com/WasifurRahman/BERT_multimodal_transformer) through the following link: 
```
pip install gdown
gdown https://drive.google.com/uc?id=12HbavGOtoVCqicvSYWl3zImli5Jz0Nou
gdown https://drive.google.com/uc?id=1VJhSc2TGrPU8zJSVTYwn5kfuG47VaNQ3. 
```


### Prerequisites:
```
* Python 3.8.10
* CUDA 11.5
* pytorch 1.12.1+cu113
* sentence-transformers 3.1.1
* transformers 4.30.2
```
**Note that the torch version can be changed to your cuda version, but please keep the transformers==4.30.2 as some functions will change in later versions**


### Pretrained model:
Downlaod the [BERT-base](https://huggingface.co/google-bert/bert-base-uncased/tree/main) , and put into directory ./BERT-EN/.
