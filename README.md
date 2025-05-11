# SAUGE

## Publication
SAUGE: Taming SAM for Uncertainty-Aligned Multi-Granularity Edge Detection ([AAAI 2025](https://doi.org/10.1609/aaai.v39i6.32615), [Arxiv](https://arxiv.org/abs/2412.12892))

Xing Liufu, Chaolei Tan, Xiaotong Lin, Yonggang Qi, Jinxuan Li, Jian-Fang Hu

## Environmental Setup
Please follow the official installation steps for SAM: https://github.com/facebookresearch/segment-anything.

## Dataset
Please follow the guide of [UAED](https://github.com/ZhouCX117/UAED_MuGE/tree/main) for preparing BSDS500 and Multicue datasets.

## Inference
Run the following command:
`python demo.py`

We have provided the checkpoint for the basic version of SAUGE (trained on the `BSDS500` dataset). Please adjust the dataset path and the checkpoint path for SAM in `demo.py` accordingly. 

## Training
Please modify the relevant data/hyperparameters and other configurations in `train.py`, and then run the following command for training:
`nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port 123 train.py > nohup.log 2>&1 &`


## Pre-computed Results and Evaluation
We have organized the precomputed results for BSDS500 datasets under various settings in the `eval_res` directory. To reproduce the results reported in the paper, modify the relevant paths and configurations in `eval_res/best_ods_ois.py` and then run `python eval_res/best_ods_ois.py`.

## Acknowledgement & Citing SAUGE

The work is highly based on the [Segment Anything](https://github.com/facebookresearch/segment-anything) and [UAED](https://github.com/ZhouCX117/UAED_MuGE/tree/main). We gratefully acknowledge their excellent work. 

If this project supports your research, please consider including a citation in your publications.

```
Liufu, X., Tan, C., Lin, X., Qi, Y., Li, J., & Hu, J.-F. (2025). SAUGE: Taming SAM for Uncertainty-Aligned Multi-Granularity Edge Detection. Proceedings of the AAAI Conference on Artificial Intelligence, 39(6), 5766-5774. https://doi.org/10.1609/aaai.v39i6.32615
```

