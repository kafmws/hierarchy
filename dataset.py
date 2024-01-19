import sys
sys.path.append('/root/projects/readings')

import os
import torch
import pickle
from typing import Callable, Iterable, Optional, Tuple, Any
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import CIFAR100, ImageNet, ImageFolder

DATA_DIR = {
    'imagenet1k': '/data/imagenet_sets/in1k/',
    'iwildcam36': '/data/wilds_sq336_36/',
    'imagenet1kfeature': '/root/projects/readings/work/feature_dataset',
    'iwildcam36feature': '/root/projects/readings/work/feature_dataset',
    
    'imagenet1kfeature_benchmark': '/root/projects/readings/work/feature_dataset_benchmark',
}


class _ImageNet(ImageNet):
    """extend `torchvision.datasets.ImageNet` return path in `__getitem__`

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports `train`, or `val`.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, `transforms.RandomCrop`
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    def __getitem__(self, index) -> Tuple[Any, Any, Any]:
        sample, target = super().__getitem__(index=index)
        path, _ = self.samples[index]
        return sample, target, path
    
    def ImageNet(split, transform):
        dataset = _ImageNet(root=DATA_DIR['imagenet1k'], split=split, transform=transform)
        return dataset


class ImageNetFeatureDataset(ImageNet):
    """Dataset of ImageNet-1k's CLIP features, modified from `torchvision.datasets.ImageNet`.
    """
    def __init__(self, split, model, arch, root=DATA_DIR['imagenet1kfeature']) -> Dataset:
        super().__init__(root=DATA_DIR['imagenet1k'], split=split)
        self.featfile = os.path.join(root, model.lower(), arch.replace('/', '-'), f'imagenet1k_{split}_features.pkl')
        assert os.path.isfile(self.featfile), f'featrue file {self.featfile} not found'
        with open(self.featfile, 'rb') as file:
            self.feature_data = pickle.load(file)
        self.feature_samples = self.feature_data['features']
        self.feature_dtype = self.feature_samples[0][2].dtype
        assert len(self.feature_samples) == len(self.samples), \
            f'number of features {len(self.feature_samples)} does not match number of images {len(self.samples)}'

    def __getitem__(self, index) -> Tuple[Any, Any, str]:
        imgpath, _target = self.samples[index]
        imgname, target, feature = self.feature_samples[index]
        assert imgname in imgpath and _target == target
        return torch.HalfTensor(feature), target, imgpath


class IWildCam(ImageFolder):
    
    def __init__(self, root: str, split: str, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None):
        super().__init__(os.path.join(root, split), transform, target_transform)
        self.split = split
    
    def __getitem__(self, index) -> Tuple[Any, Any, Any]:
        sample, target = super().__getitem__(index=index)
        path, _ = self.samples[index]
        return sample, target, path
    
    def IWildCam(split, transform, root):
        dataset = IWildCam(root=root, split=split, transform=transform)
        return dataset


class IWildCamFeatureDataset(IWildCam):
    """Dataset of ImageNet-1k's CLIP features, modified from `torchvision.datasets.ImageNet`.
    """
    def __init__(self, split, model, arch, root) -> Dataset:
        super().__init__(root=DATA_DIR['iwildcam36'], split=split)
        self.featfile = os.path.join(root, model.lower(), arch.replace('/', '-'), f'iwildcam36_{split}_features.pkl')
        assert os.path.isfile(self.featfile), f'featrue file {self.featfile} not found'
        with open(self.featfile, 'rb') as file:
            self.feature_data = pickle.load(file)
        self.feature_samples = self.feature_data['features']
        self.feature_dtype = self.feature_samples[0][2].dtype
        assert len(self.feature_samples) == len(self.samples), \
            f'number of features {len(self.feature_samples)} does not match number of images {len(self.samples)}'

    def __getitem__(self, index) -> Tuple[Any, Any, str]:
        imgpath, _target = self.samples[index]
        imgname, target, feature = self.feature_samples[index]
        assert imgname in imgpath and _target == target
        return torch.HalfTensor(feature), torch.LongTensor([target]), imgpath


DATASET = {
    'imagenet1k': _ImageNet.ImageNet,
    'iwildcam36': IWildCam.IWildCam,
}

FEAT_DATASET = {
    'imagenet1k': ImageNetFeatureDataset,
    'iwildcam36': IWildCamFeatureDataset,
}


def get_dataset(dataset_name, split, transform, root=None):
    if root is None:
        root = DATA_DIR['iwildcam36']
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
    
    def test_iwildcam36():
        from clip_analysis import preprocess
        dataset = get_dataset(dataset_name='iwildcam36', split='train', transform=preprocess)
        print(dataset[0])
    
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
    ds_feature = get_feature_dataset(dataset_name='iwildcam36', split='test', model='clip', arch='ViT-L/14@336px')
    print(ds_feature[0])
