# Readme
## Preparation

pull CLIP repo.

```
.
├── clip -> ../CLIP/clip                           # git clone https://github.com/openai/CLIP.git
├── dataset
│   └── iwildcam36
├── feature_dataset
│   └── openai_clip
│       └── ViT-L-14@336px
└── hierarchical
    └── iwildcam36
```

## Dataset

https://drive.google.com/file/d/1qj2OQxumv7lv8KBQRdxGG3lRZ6CFJ3ES/view?usp=sharing

## Environment

```
python                    3.11
CUDA                      11.7
torch                     2.0.0
transformers              4.35.2
nvidia-cuda-cupti-cu11    11.7.101
nvidia-cuda-nvrtc-cu11    11.7.99
nvidia-cuda-runtime-cu11  11.7.99
```


## Reproduction
```
python clip_analysis.py   # for all experiments in the paper, read the code
```

