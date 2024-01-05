import sys
sys.path.extend(['/root/projects/readings/CLIP', '/root/projects/readings'])

import os
import torch
import pickle
from tqdm import tqdm
from CLIP.clip import clip
from typing import Tuple, Any
from torch.utils.data import DataLoader

from utils import set_seed
from dataset import DATASET, FEAT_DATASET
from prompts import clsname2prompt

# config
seed = 42
model_name = 'ViT-L/14'
datasets = {'imagenet1k': ['train', 'val']}
output_dir = '/root/projects/readings/work/feature_dataset'

for ds in datasets:
    assert ds in DATASET, f'{ds} should be registered entry in __DATASET__'
set_seed(42)

# Load the model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load(model_name, device)


def collect_image_features():
    
    for dataset_name, splits in datasets.items():
        for split in splits:
            
            feat_output_dir = f'{output_dir}/clip/{model_name.replace("/", "-")}'
            os.makedirs(feat_output_dir, exist_ok=True)
            
            print(f'preparing {dataset_name} {split} features...')
            dataset = DATASET[dataset_name](split, preprocess)
            loader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=2)

            sample_features = []  # [path, target, feature]
            path_prefix_len = len(dataset.root) + len(dataset.split) + len('/')

            # calculate features
            with torch.no_grad():
                for images, targets, paths in tqdm(loader, ncols=60, dynamic_ncols=True):
                    images = images.to(device)
                    image_features = model.encode_image(images)
                    for path, target, feature, in zip(paths, targets, image_features):
                        sample_features.append((path[path_prefix_len:], target.item(), feature.cpu().numpy()))
            
            data = {
                'fname': f'{dataset_name}_{split}_features.pkl',
                'features': sample_features,
                'transform': preprocess,
                'model': model_name,
            }

            # torch.save(obj=data, f=os.path.join(feat_output_dir, data['fname'] + '.pth'))
            with open(os.path.join(feat_output_dir, data['fname']), 'wb') as file:
                pickle.dump(obj=data, file=file)


def collect_clip_logits():
    
    dataset_name = 'imagenet1k'
    dataset = FEAT_DATASET[dataset_name](split='val', arch='clip', model='ViT-L/14')
    split = dataset.split
    loader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=2)
    print(f'preparing {dataset_name} {split} logits...')
    
    feat_output_dir = f'{output_dir}/clip/{model_name.replace("/", "-")}'
    os.makedirs(feat_output_dir, exist_ok=True)
    
    sample_logits = []  # [path, target, feature]
    path_prefix_len = len(dataset.root) + len(dataset.split) + len('/')

    # texts = [f'a photo of a {", ".join(c)}' for c in dataset.classes]                                                             # ViT-L/14
    # text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(device)                                                         # accuracy:  70.86
    # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c[0]}") for c in dataset.classes]).to(device)                          # accuracy:  70.20
    # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in dataset.classes]).to(device)                             # accuracy:  70.82
    # text_inputs = torch.cat([clip.tokenize([f'a photo of a {", or ".join(c)}' for c in dataset.classes])]).to(device)             # accuracy:  71.36
    text_inputs = clsname2prompt(dataset.classes)

    for i, texts in enumerate(text_inputs):
        # calculate text features
        
        if len(texts) > 1000:
            mutiple = len(texts) / 1000
            
            global device
            text_features = []
            with torch.no_grad():
                texts = clip.tokenize(texts).to(device)  # tokenize
                _batch = list(range(0, len(texts), 256))
                if _batch[-1] < len(texts):
                    _batch.append(len(texts))
                for endi in range(1, len(_batch)):
                    text_features.extend(model.encode_text(texts[_batch[endi - 1]:_batch[endi]]))
            text_features = torch.stack(text_features, dim=0)
            assert text_features.shape[0] == len(texts), f'{text_features.shape[0]} != {len(texts)}'
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            correct = 0
            with torch.no_grad():
                for images, targets, paths in tqdm(loader, ncols=60, dynamic_ncols=True):
                    targets: torch.Tensor = targets.to(device)
                    image_features: torch.Tensor = images.to(device)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    # image_features = image_features.cpu().numpy()
                    logits = image_features @ text_features.T
                    # logits = torch.Tensor(logits)
                    similarity = (100.0 * logits).softmax(dim=-1)
                    for path, target, _logits, in zip(paths, targets, logits):
                        sample_logits.append((path[path_prefix_len:], target.item(), _logits.cpu().numpy()))
                    predicts = similarity.topk(1, dim=1).indices.squeeze(dim=1)
                    predicts = (predicts / mutiple)
                    
                    correct += sum(torch.eq(targets, predicts)).item()
        
        else:
            with torch.no_grad():
                texts = clip.tokenize(texts).to(device)  # tokenize
                text_features: torch.Tensor = model.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            correct = 0
            with torch.no_grad():
                for images, targets, paths in tqdm(loader, ncols=60, dynamic_ncols=True):
                    targets: torch.Tensor = targets.to(device)
                    image_features: torch.Tensor = images.to(device)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    logits = image_features @ text_features.T
                    similarity = (100.0 * logits).softmax(dim=-1)
                    for path, target, _logits, in zip(paths, targets, logits):
                        sample_logits.append((path[path_prefix_len:], target.item(), _logits.cpu().numpy()))
                    predicts = similarity.topk(1, dim=1).indices.squeeze(dim=1)
                    
                    correct += sum(torch.eq(targets, predicts)).item()
        
        print(f'accuracy of prompt #{i}: {correct / len(dataset) * 100: .2f}')
        # data = {
        #     'fname': f'{dataset_name}_{split}_logits.pkl',
        #     'text_features': text_features,
        #     'logits': sample_logits,
        #     'model': model_name,
        # }

        # # torch.save(obj=data, f=os.path.join(feat_output_dir, data['fname'] + '.pth'))
        # with open(os.path.join(feat_output_dir, data['fname']), 'wb') as file:
        #     pickle.dump(obj=data, file=file)


if __name__ == '__main__':
    # collect_image_features()
    collect_clip_logits()
    

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
