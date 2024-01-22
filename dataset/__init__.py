import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Iterable
from torch.utils.data import ConcatDataset

DATA_DIR = {
    'imagenet1k': '/data/imagenet_sets/in1k/',
    'iwildcam36': '/data/wilds_sq336_36/',
    'animal90': '/data/kafm/vision_datasets/animal90/',
    'imagenet1kfeature': '/root/projects/readings/work/feature_dataset',
    'iwildcam36feature': '/root/projects/readings/work/feature_dataset',
    'animal90feature': '/root/projects/readings/work/feature_dataset',
    'imagenet1kfeature_benchmark': '/root/projects/readings/work/feature_dataset_benchmark',
}

from dataset.animal90 import Animal, AnimalFeatureDataset
from dataset.iwildcam36 import IWildCam, IWildCamFeatureDataset
from dataset.imagenet1k import _ImageNet, ImageNetFeatureDataset

DATASET = {
    'imagenet1k': _ImageNet.ImageNet,
    'iwildcam36': IWildCam.IWildCam,
    'animal90': Animal.Animal,
}

FEAT_DATASET = {
    'imagenet1k': ImageNetFeatureDataset,
    'iwildcam36': IWildCamFeatureDataset,
    'animal90': AnimalFeatureDataset,
}


def get_dataset(dataset_name, split, transform, root=None):
    if root is None:
        root = DATA_DIR[dataset_name]
    dataset = DATASET[dataset_name](root=root, split=split, transform=transform)
    dataset.split = split
    dataset.transform = transform
    dataset.dataset_name = dataset_name
    return dataset


def get_feature_dataset(dataset_name, split, model, arch, root=None):
    if root is None:
        root = DATA_DIR[f'{dataset_name}feature']
    if not isinstance(split, str) and isinstance(split, Iterable):
        datasets = [get_feature_dataset(dataset_name, s, model, arch, root) for s in split]
        dataset = ConcatDataset(datasets=datasets)
    else:
        dataset = FEAT_DATASET[dataset_name](root=root, split=split, model=model, arch=arch)
    dataset.split = split
    dataset.dataset_name = dataset_name
    dataset.featurizer = f'{model}:{arch}'
    return dataset


if __name__ == '__main__':
    from clip_analysis import preprocess

    def test_iwildcam36():
        dataset = get_dataset(dataset_name='iwildcam36', split='test', transform=preprocess)
        print(dataset[0])
        print(f'len: {len(dataset)}')

    def test_animal90():
        dataset = get_dataset(dataset_name='animal90', split='test', transform=preprocess)
        print(dataset[0])
        print(f'len: {len(dataset)}')

    def test_feauture_dataset():
        ds_feature = get_feature_dataset(dataset_name='imagenet1k', split='val', model='clip', arch='ViT-B/32')
        with open('wnid2idx.txt', 'w') as file:
            for i, wnid in enumerate(ds_feature.wnids):
                file.write(f'{i}\t{wnid}\n')

        print(ds_feature[0])
        for i, cls in enumerate(ds_feature.classes):
            print(f'{i}: {cls} {ds_feature.wnids[i]}')

        # print(f'wnid: {ds_feature.wnids[657], ds_feature.classes[657]}')  # 657: ('missile',)
        # print(f'wnid: {ds_feature.wnids[744], ds_feature.classes[744]}')  # 744: ('projectile', 'missile')

        # print(f'wnid: {ds_feature.wnids[836], ds_feature.classes[836]}')  # 836: ('sunglass',)
        # print(f'wnid: {ds_feature.wnids[837], ds_feature.classes[837]}')  # 837: ('sunglasses', 'dark glasses', 'shades')

        print(f'wnid: {ds_feature.wnids[745], ds_feature.classes[745]}')

    test_iwildcam36()
    # test_feauture_dataset()
    # ds_feature = get_feature_dataset(dataset_name='iwildcam36', split='test', model='clip', arch='ViT-L/14@336px')
    # print(ds_feature[0])

    # test_animal90()
