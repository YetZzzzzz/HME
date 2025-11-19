# HME: Hyper-Modality Enhancement for Multimodal Sentiment Analysis with Missing Modalities
We propose the Hyper-Modality Enhancement for Multimodal Sentiment Analysis with Missing Modalities (HME) to address the modality missingness in real-world scenarios, which is accepted by [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/116212).

### The Framework of HME:
![image](https://github.com/YetZzzzzz/HME/blob/main/framework.png)
Figure 1: Visualization of HME framework. It consists of three modules: Pre-Processing Module, Hyper-Modality Representation Generation Module and Multimodal Hyper-Modality Fusion Module.

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
Downlaod the [BERT-base](https://huggingface.co/google-bert/bert-base-uncased/tree/main) , and put into directory ./BERT_en/.

### Datasets:
**Please move the following datasets into directory ```./datasets/```**

The aligned CMU-MOSI and CMU-MOSEI datasets can be downloaded according to [DiCMoR](https://github.com/mdswyz/DiCMoR) and [IMDer](https://github.com/mdswyz/IMDer), rename the pkl as ```aligned_{dataset}.pkl```. 

### Run HME
For MOSI and MOSEI dataset, please run the following code in ```./HME_MSA/``` through:
```
python3 HME_main.py --dataset='mosi' --learning_rate=2e-5 --d_l=192 --missing_rate=0.2 --layers=4 --hyper_depth=3 --latent_layers=4 --latent_dim=192 --n_epochs=100
python3 HME_main.py --dataset='mosei' --learning_rate=2e-5 --d_l=192 --missing_rate=0.1 --layers=2 --hyper_depth=3 --latent_layers=3 --latent_dim=192 --n_epochs=100
```
Here ```missing_rate``` denotes the MR value.

### Citation:
Please cite our paper if you find our work useful for your research:
```
@inproceedings{zhuanghyper,
  title={Hyper-Modality Enhancement for Multimodal Sentiment Analysis with Missing Modalities},
  author={Zhuang, Yan and Liu, Minhao and Bai, Wei and Zhang, Yanru and Li, Wei and Deng, Jiawen and Ren, Fuji},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

### Acknowledgement
Thanks to [DiCMoR](https://github.com/mdswyz/DiCMoR), [IMDer](https://github.com/mdswyz/IMDer), [GCNet](https://github.com/zeroQiaoba/GCNet), [LNLN](https://github.com/Haoyu-ha/LNLN) and [HKT](https://github.com/matalvepu/HKT) for their great help to our codes and research. 
