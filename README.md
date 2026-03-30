# Multimodal Competitive Fusion Model (MCFM)
Source code (PyTorch) of the paper "MCFM: Multimodal competitive fusion mechanism for sentiment analysis", which is accepted by The IEEE Transactions on Computational Social Systems.


DOI:10.1109/TCSS.2025.3598637

Paper Link: https://ieeexplore.ieee.org/document/11165416
## Environment setup
1. create a new environment using conda or pip
2. ```pip install -r requirements.txt```


## Download Data
The three datasets (CMU-MOSI, CMU-MOSEI, and CH-SIMS) are available from this link: https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk

## Data Directory
To run our preprocessing and training codes directly, please put the necessary files from downloaded data in separate folders as described below.

```
/data/
    mosi/
        raw/
        label.csv
    mosei/
        raw/
        label.csv
    sims/
        raw/
        label.csv
```

## Audio Extraction
Before running the training code, please extract the audio from the raw video files.

```
python extract_audio.py

options:
  --dataset DATASET     dataset name (mosi, mosei, or sims) (default: mosi)
```
```
python run.py

options (optional):
  --seed SEED           random seed (default: 1)
  
  --batch_size BATCH_SIZE
                        batch size (default: 8)
                        
  --lr LR               learning rate (default: 5e-6, recommended: 5e-6 for mosi, mosei, 1e-5 for sims)
                        
  --dataset DATASET     dataset name (mosi, mosei, or sims) (default: mosi)
  
  --num_hidden_layers NUM_HIDDEN_LAYERS
                        number of hidden layers for cross-modality encoder (default: 5)

```
# Cite
If you use MCFM in a scientific publication, we would appreciate citations to the following paper:
@ARTICLE{11165416,
  author={Xing, Mali and Zhai, Zilang and Deng, Muqing and Cai, Qianqian and Ren, Hongru and Wang, Tian},
  journal={IEEE Transactions on Computational Social Systems}, 
  title={MCFM: Multimodal Competitive Fusion Mechanism for Sentiment Analysis}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  keywords={Sentiment analysis;Reliability;Feature extraction;Transformers;Semantics;Tensors;Visualization;Accuracy;Reliability engineering;Mutual information;Competition mechanism;feature decomposition;multimodal sentiment analysis (MSA);orthogonal channel attention (OCA)},
  doi={10.1109/TCSS.2025.3598637}}


Copyright © 2026 [weedy3000]. All rights reserved.


