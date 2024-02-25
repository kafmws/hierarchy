import os
import sys
import timeit

# for PYTHONPATH
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.extend([project_root])
os.chdir(project_root)

# for reproducibility
from utils import set_seed

seed = 42
set_seed(seed)

import torch
import numpy as np
from tqdm import tqdm
import seaborn as sns
from itertools import accumulate
from matplotlib import pyplot as plt

from models import get_model
from clip_analysis import encode_text_batch
from dataset import get_dataset, get_feature_dataset
from prompts import clsname2prompt, hierarchical_prompt
from hierarchical.hierarchy import get_hierarchy, Hierarchy


dataset_settings = {
    'iwildcam36': [
        [0, 4],
    ],
    'aircraft': [[0, 2], [1, 2]],
    'animal90': [[0, 2], [1, 2]],
}
model_settings = [
    ('openai_clip', 'ViT-L/14@336px'),
    ('eva_clip', 'EVA02-CLIP-L-14-336'),
    ('eva_clip', 'EVA02-CLIP-bigE-14-plus'),
]
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'


def draw_text_similiarity(model_name, arch, datasets):
    # model_name, arch = 'openai_clip', 'ViT-L/14@336px'
    model, tokenizer, _ = get_model(model_name, arch, device)
    arch = arch.replace('/', '-')

    for dataset, selected_layers in datasets.items():
        h: Hierarchy = get_hierarchy(dataset)
        text_inputs, (isa_mask, layer_isa_mask, layer_mask), hierarchical_targets, classnames, h = hierarchical_prompt(dataset)
        layer_cnt, layer2name = h.layer_cnt, h.layer2name
        isa_mask = torch.from_numpy(isa_mask).int().to(device)
        layer_mask = torch.from_numpy(layer_mask).int().to(device)
        layer_isa_mask = torch.from_numpy(layer_isa_mask).int().to(device)
        flat_classnames = [item for sublist in classnames for item in sublist]
        layer_offset = [0] + list(accumulate(layer_cnt))
        n_class = sum(layer_cnt)

        texts = text_inputs[0]
        text_features = encode_text_batch(
            model, tokenizer, texts=texts, n_classes=layer_offset[-1], text_fusion=False, device=device
        )

        # text_dim = text_features.shape[1]
        # isa_mask = torch.stack([isa_mask[i].reshape(101, 1).expand(101, text_dim) for i in range(n_class)])

        for selected_layer in selected_layers:
            for cur, next in zip(selected_layer[:-1], selected_layer[1:]):
                nodes = h.get_layer_nodes(cur)
                m = np.zeros((len(nodes), len(nodes)))
                for node_a in nodes:
                    for node_b in nodes:
                        mask = isa_mask[node_b.id]
                        mask = mask & layer_mask[next]
                        num = int(sum(mask))
                        # group = mask.reshape(101, 1).expand(101, text_features.shape[1]) * text_features
                        # group = mask * text_features  # change isa_mask
                        group = text_features[torch.where(mask != 0)]
                        similarity = group @ text_features[node_a.id]
                        mean_sim = sum(similarity) / num
                        m[node_a.inlayer_idx][node_b.inlayer_idx] = mean_sim.item()

                # draw
                plt.clf()
                labels = [node.name for node in nodes]
                heatmap = sns.heatmap(m, cmap='Blues', annot=len(nodes) <= 15, fmt='.2f')
                heatmap.set_xticklabels(labels, rotation=-45, ha='right')
                heatmap.set_yticklabels(labels, rotation=-45, ha='right')
                heatmap.set_title(f'Label similarity between {h.layer2name[cur]} and {h.layer2name[next]}')
                plt.savefig(f'pic/text/{dataset}_{model_name}_{arch}_{h.layer2name[cur]}_{h.layer2name[next]}.png', dpi=300)


for model_name, arch in model_settings:
    draw_text_similiarity(
        model_name,
        arch,
        dataset_settings,
    )
