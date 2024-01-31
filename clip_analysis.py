import os
import sys

# for PYTHONPATH
sys.path.extend(['/root/projects/readings', '/root/projects/readings/work'])

# for reproducibility
from utils import set_seed

seed = 42
set_seed(seed)

import torch
import pickle
import numpy as np
from tqdm import tqdm
from CLIP.clip import clip
from itertools import accumulate
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from typing import Counter, List, Tuple, Any, Iterable
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from hierarchical.hierarchy import get_hierarchy
from dataset import get_dataset, get_feature_dataset
from prompts import clsname2prompt, hierarchical_prompt, iwildcam36_prompts

# config
model_name = 'clip'
arch = 'ViT-L/14@336px'
# datasets = {'imagenet1k': ['val']}
# datasets = {'iwildcam36': ['train']}
# datasets = {'animal90': ['test']}
datasets = {'aircraft': ['test']}
output_dir = '/root/projects/readings/work'

feat_output_dir = os.path.join(output_dir, 'feature_dataset', model_name, arch.replace("/", "-"))
os.makedirs(feat_output_dir, exist_ok=True)

pic_output_dir = os.path.join(output_dir, 'pic')
os.makedirs(pic_output_dir, exist_ok=True)

# Load the model
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load(arch, device)


def collect_image_features():
    for dataset_name, splits in datasets.items():
        for split in splits:
            print(f'preparing {dataset_name} {split} features...')
            dataset = get_dataset(dataset_name=dataset_name, split=split, transform=preprocess)
            loader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=2)

            sample_features = []  # [path, target, feature]
            path_prefix_len = len(dataset.root) + len(dataset.split) + len('/')

            # calculate features
            with torch.no_grad():
                for images, targets, paths in tqdm(loader, ncols=60, dynamic_ncols=True):
                    images = images.to(device)
                    image_features = model.encode_image(images)
                    for (
                        path,
                        target,
                        feature,
                    ) in zip(paths, targets, image_features):
                        sample_features.append((path[path_prefix_len:], target.item(), feature.cpu().numpy()))

            data = {
                'fname': f'{dataset_name}_{split}_features.pkl',
                'model': f'{model_name}:{arch}',
                'features': sample_features,
                'transform': preprocess,
            }

            # torch.save(obj=data, f=os.path.join(feat_output_dir, data['fname'] + '.pth'))
            with open(os.path.join(feat_output_dir, data['fname']), 'wb') as file:
                pickle.dump(obj=data, file=file)


def encode_text_batch(texts: List[str], n_classes, text_fusion, device):
    multiple = len(texts) / n_classes
    assert multiple == int(multiple)
    multiple = int(multiple)

    text_features = []
    # calculate text features
    with torch.no_grad():
        tokens = clip.tokenize(texts).to(device)  # tokenize
        _batch = list(range(0, len(tokens), multiple))
        if _batch[-1] < len(tokens):
            _batch.append(len(tokens))
        for endi in range(1, len(_batch)):
            batch_text_features = model.encode_text(tokens[_batch[endi - 1] : _batch[endi]])
            batch_text_features /= batch_text_features.norm(dim=-1, keepdim=True)
            if text_fusion:
                batch_text_features = batch_text_features.mean(dim=0)
                batch_text_features /= batch_text_features.norm()
                batch_text_features = [batch_text_features]
            text_features.extend(batch_text_features)
    text_features = torch.stack(text_features, dim=0)
    assert (
        text_features.shape[0] == n_classes if text_fusion else len(tokens)
    ), f'{text_features.shape[0]} != {n_classes if text_fusion else len(tokens)}'
    return text_features


def collect_clip_logits(text_fusion=False, only: int = None, dump=False):
    """CLIP zero-shot inference with multiple prompts.

    Args:
        text_fusion (bool, optional): To mean text prompt embedding or not. Defaults to False.
        only (int, optional): Test all kinds of prompts or which only. Defaults to None (test all).
        dump (bool, optional): Whether save the logits or not. Defaults to True.
    """

    dataset = get_feature_dataset(dataset_name='imagenet1k', split='val', model=model_name, arch=arch)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=2)
    print(f'preparing {dataset.dataset_name} {dataset.split} logits...')

    sample_logits = []  # [path, target, feature]
    path_prefix_len = len(dataset.root) + len(dataset.split) + len('/')

    # best prompt #3 w/o ensembling, #4 w/ ensembling
    n_classes = len(dataset.classes)
    text_inputs = clsname2prompt(dataset.dataset_name, dataset.classes)
    if only:
        text_inputs = text_inputs[only : only + 1]

    for i, texts in enumerate(text_inputs):
        text_features = encode_text_batch(texts=texts, n_classes=n_classes, text_fusion=text_fusion, device=device)
        # if not len(texts) > n_classes and not text_fusion:  # multiple == 1 also performs norm
        #     text_features /= text_features.norm(dim=-1, keepdim=True)

        correct = 0
        with torch.no_grad():
            for images, targets, paths in tqdm(loader, ncols=60, dynamic_ncols=True):
                targets: torch.Tensor = targets.to(device)
                image_features: torch.Tensor = images.to(device)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_features.T
                similarity = (100.0 * logits).softmax(dim=-1)
                for (
                    path,
                    target,
                    _logits,
                ) in zip(paths, targets, logits):
                    sample_logits.append((path[path_prefix_len:], target.item(), _logits.cpu().numpy()))
                predicts = similarity.topk(1, dim=1).indices.squeeze(dim=1)
                if (
                    not text_fusion and len(texts) > n_classes
                ):  # multiple template for each class and not fuse the text embedding
                    predicts = predicts / (len(texts) / n_classes)

                """ correct two `sunglasses` classes, which trivially contributes to 0.06% accuracy"""
                # predicts = torch.where(predicts == 836, 1000, predicts)
                # predicts = torch.where(predicts == 837, 1000, predicts)
                # targets = torch.where(targets == 836, 1000, targets)
                # targets = torch.where(targets == 837, 1000, targets)

                correct += sum(torch.eq(targets, predicts)).item()

        print(f'accuracy of prompt #{i}: {correct / len(dataset) * 100: .2f}')

        if dump:
            data = {
                'fname': f'{dataset.dataset_name}_{dataset.split}_logits.pkl',
                'model': f'{model_name}:{arch}',
                'text_features': text_features,
                'logits': sample_logits,
                'texts': texts,
                'seed': seed,
            }

            with open(os.path.join(feat_output_dir, data['fname']), 'wb') as file:
                pickle.dump(obj=data, file=file)


def hierarchical_inference(dataset, selected_layers=None, detailed=False):
    feat_output_dir = f'{output_dir}/{model_name}/{arch.replace("/", "-")}'
    os.makedirs(feat_output_dir, exist_ok=True)

    sample_logits = []  # [path, target, feature]
    path_prefix_len = len(dataset.root) + len(dataset.split) + len('/')

    text_inputs, pointers, layer_mask, hierarchical_targets, classnames, h = hierarchical_prompt(dataset.dataset_name)
    layer_cnt, layer2name = h.layer_cnt, h.layer2name
    for i in range(len(pointers)):
        for j in range(len(pointers[i])):
            pointers[i][j] = pointers[i][j].to(device)
    layer_mask = torch.from_numpy(layer_mask).to(device)
    layer_offset = [0] + list(accumulate(layer_cnt))

    row = arch[:-2] + (' HI' if selected_layers else ' ZS') + '     &' + '&'.join(['{:^9}'] * len(layer_cnt)) + '\\\\'

    for i, texts in enumerate(text_inputs):
        print(f'for prompts {i}#')

        text_features = encode_text_batch(texts=texts, n_classes=layer_offset[-1], text_fusion=False, device=device)

        # collect all results
        if not selected_layers:
            correct = [0] * len(layer_cnt)
            preds = [[] for _ in range(len(layer_cnt))]
            labels = [[] for _ in range(len(layer_cnt))]
        else:
            correct = [[0] * len(layer_cnt) for _ in range(len(selected_layers))]
            preds = [[[] for _ in range(len(layer_cnt))] for _ in range(len(selected_layers))]
            labels = [[[] for _ in range(len(layer_cnt))] for _ in range(len(selected_layers))]

        with torch.no_grad():
            # for images, targets, paths in tqdm(dataset, ncols=60, dynamic_ncols=True):
            for images, targets, paths in dataset:
                # layerify target
                layer_targets = hierarchical_targets[targets]
                targets = torch.LongTensor(layer_targets)

                targets: torch.Tensor = targets.to(device)
                image_features: torch.Tensor = images.to(device)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_features.T
                similarity = (100.0 * logits).softmax(dim=-1)
                # for path, target, _logits, in zip(paths, targets, logits):
                #     sample_logits.append((path[path_prefix_len:], target.item(), _logits.cpu().numpy()))

                # layer-by-layer matching
                if selected_layers:
                    for i, selected_layer in enumerate(selected_layers):
                        for layer in selected_layer:
                            if layer == selected_layer[0]:
                                mask = layer_mask[layer]
                            else:
                                mask = pointers[predicts.item()][layer]
                            predicts = (similarity * mask).topk(1, dim=-1).indices
                            correct[i][layer] += sum(torch.eq(targets[layer], predicts)).item()

                            preds[i][layer].append(predicts.item() - layer_offset[layer])
                            labels[i][layer].append(targets[layer].item() - layer_offset[layer])
                else:
                    for layer in range(0, len(layer_cnt)):
                        masked_similarity = similarity * layer_mask[layer]
                        predicts = masked_similarity.topk(1, dim=-1).indices
                        correct[layer] += sum(torch.eq(targets[layer], predicts)).item()

                        preds[layer].append(predicts.item() - layer_offset[layer])
                        labels[layer].append(targets[layer].item() - layer_offset[layer])

        # latex accuracy output
        if selected_layers:
            for i, selected_layer in enumerate(selected_layers):
                res = [
                    '{:.2f}'.format(correct[i][idx] * 100 / len(dataset)) if idx in selected_layer else '-'
                    for idx in range(len(layer_cnt))
                ]
                print(row.format(*res))
        else:
            res = ['{:.2f}'.format(correct[idx] * 100 / len(dataset)) for idx in range(len(layer_cnt))]
            print(row.format(*res))

        # detailed results
        if detailed:
            if selected_layers:
                for i, selected_layer in enumerate(selected_layers):
                    pass
            else:
                for layer in range(0, len(layer_cnt)):
                    report = classification_report(y_true=labels[layer], y_pred=preds[layer], target_names=classnames[layer])
                    # print(report)

                    # debug
                    diff = np.where(np.array(labels[layer]) != np.array(preds[layer]))
                    errs = np.array(dataset.targets)[diff]
                    errs = list(map(lambda idx: dataset.classes[idx], errs))
                    cnt = list(Counter(errs).items())
                    cnt.sort(key=lambda item: item[1], reverse=True)
                    print(cnt)

                    cm = confusion_matrix(y_true=labels[layer], y_pred=preds[layer], normalize='true')
                    disp = ConfusionMatrixDisplay(cm, display_labels=classnames[layer])
                    disp.plot(cmap='Blues', values_format='.2%', text_kw={'size': 6})
                    figpath = os.path.join(pic_output_dir, f'{dataset.dataset_name}_{layer2name[layer]}.png')
                    title = f'confusion matrix of {layer2name[layer]} layer'
                    plt.title(label=title)
                    plt.tight_layout()
                    plt.savefig(figpath, dpi=300)


if __name__ == '__main__':
    clip_transform = Compose([torch.HalfTensor])
    clip_target_transform = Compose([torch.as_tensor])

    def test_iwildcam36():
        dataset = get_feature_dataset(dataset_name='iwildcam36', split='test', model=model_name, arch=arch)
        print(f'loading {dataset.dataset_name} {dataset.split} feature...')

        hierarchical_inference(dataset)
        hierarchical_inference(
            dataset=dataset,
            selected_layers=[
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4],
                [0, 2, 3, 4],
                [0, 1, 3, 4],
                [0, 1, 2, 4],
                [0, 1, 4],
                [0, 2, 4],
                [0, 3, 4],
                [2, 3, 4],
                [0, 4],
                [1, 4],
                [2, 4],
                [3, 4],
                [4],
            ],
        )

    def test_animal90():
        dataset = get_feature_dataset(
            dataset_name='animal90',
            split='test',
            model=model_name,
            arch=arch,
            transform=clip_transform,
            target_transform=clip_target_transform,
        )
        print(f'loading {dataset.dataset_name} {dataset.split} feature...')

        hierarchical_inference(dataset)

    def test_aircraft():
        dataset = get_feature_dataset(
            dataset_name='aircraft',
            split='test',
            model=model_name,
            arch=arch,
            transform=clip_transform,
            target_transform=clip_target_transform,
        )
        print(f'loading {dataset.dataset_name} {dataset.split} feature...')

        hierarchical_inference(dataset)
        hierarchical_inference(
            dataset=dataset,
            selected_layers=[
                [0, 1, 2],
                [0, 1],
                [0, 2],
                [1, 2],
                [2],
            ],
        )

    # test_iwildcam36()
    # test_animal90()
    test_aircraft()

    # collect_image_features()
    # collect_clip_logits(text_fusion=True, dump=False)
    # collect_clip_logits(text_fusion=True, only=3, dump=False)


def benchmark_torchsave_pickledump():
    # comparsion between torch.load vs. pickle.load
    # torch loading time:  16.87s
    # pickle loading time:  3.20s
    import timeit

    start_time = timeit.default_timer()
    data = torch.load('work/feature_dataset/clip/ViT-B-32_pre/imagenet1k_val_features.pth')
    data = torch.load('work/feature_dataset/clip/ViT-B-32_pre/imagenet1k_train_features.pth')
    end_time = timeit.default_timer()
    print(f'torch loading time: {end_time - start_time :.2s}s')

    start_time = timeit.default_timer()
    with open('work/feature_dataset/clip/ViT-B-32/imagenet1k_val_features.pkl', 'rb') as file:
        data = pickle.load(file)
    with open('work/feature_dataset/clip/ViT-B-32/imagenet1k_train_features.pkl', 'rb') as file:
        data = pickle.load(file)
    end_time = timeit.default_timer()
    assert data
    print(f'pickle loading time: {end_time - start_time :.2s}s')
