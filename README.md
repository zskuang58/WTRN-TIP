# WTRN-TIP
Official PyTorch implementation of our TIP paper Wavelet-Based Texture Reformation Network for
Image Super-Resolution

## Dependencies

* python >= 3.7
* python packages: `pip install opencv-python imageio tensorboard tqdm`
* pytorch >= 1.1.0
* torchvision >= 0.4.0

## Prepare Dataset 

1. Download [CUFED train set](https://drive.google.com/drive/folders/1hGHy36XcmSZ1LtARWmGL5OK1IUdWJi3I) and [CUFED test set](https://drive.google.com/file/d/1Fa1mopExA9YGG1RxrCZZn7QFTYXLx6ph/view)
1. Place the datasets in this structure:
    ```
    CUFED
    ├── train
    │   ├── input
    │   └── ref 
    └── test  
    ```

## Dowmload codes

1. Clone this repo
    ```
    git clone https://github.com/zskuang58/WTRN-TIP.git
    cd WTRN-TIP
    ```

## Evaluation

1. Prepare pre-trained models and modify "model_path" in eval.sh

1. Prepare CUFED dataset and modify "dataset_dir" in eval.sh

1. Run evaluation
    ```
    sh eval.sh
    ```
1. Evaluation results are in "save_dir" (default: `./eval/CUFED/WTRN`)

## Training

1. Prepare CUFED dataset and modify "dataset_dir" in train.sh
1. Run training
    ```
    sh train.sh
    ```
1. The training results are in the "save_dir" (default: `./train/CUFED/WTRN`)

## Acknowledgement

We borrow some codes from [TTSR](https://github.com/researchmm/TTSR) and [WCT2](https://github.com/clovaai/WCT2). We thank the authors for their great work.

## Citation

Please consider citing our paper in your publications if it is useful for your research.
```
### todo

```