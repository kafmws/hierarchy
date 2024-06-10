# Preparation

pull CLIP and EVA-CLIP repo.

```
.
├── clip -> ../CLIP/clip                           # git clone https://github.com/openai/CLIP.git
├── dataset
│   └── iwildcam36
├── doc
├── eva_clip -> ../EVA/EVA-CLIP/rei/eva_clip       # git clone https://github.com/baaivision/EVA.git
├── feature_dataset
│   ├── eva_clip
│   │   ├── EVA01-CLIP-g-14
│   │   ├── EVA02-CLIP-bigE-14-plus
│   │   ├── EVA02-CLIP-g-14
│   │   └── EVA02-CLIP-L-14-336
│   ├── eva_clip_8B
│   │   └── BAAI-EVA-CLIP-8B
│   └── openai_clip
│       └── ViT-L-14@336px
└── hierarchical
    └── iwildcam36
```

# download dataset

https://drive.google.com/file/d/1qj2OQxumv7lv8KBQRdxGG3lRZ6CFJ3ES/view?usp=sharing

# environment
python                    3.11
CUDA                      11.7
torch                     2.0.0
transformers              4.35.2
nvidia-cuda-cupti-cu11    11.7.101
nvidia-cuda-nvrtc-cu11    11.7.99
nvidia-cuda-runtime-cu11  11.7.99

# run
python clip_analysis.py   # for all experiments in the paper
