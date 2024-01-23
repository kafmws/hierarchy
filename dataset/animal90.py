import os
import torch
import pickle
from torch.utils.data import Dataset
from typing import Callable, Tuple, Any
from torchvision.datasets import ImageFolder

from dataset import DATA_DIR


class Animal(ImageFolder):
    """dataset class for animal90."""

    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
    ):
        super().__init__(os.path.join(root, split), transform, target_transform)
        self.split = split
        self.n_classes = len(self.classes)

    def __getitem__(self, index) -> Tuple[Any, Any, Any]:
        sample, target = super().__getitem__(index=index)
        path, _ = self.samples[index]
        return sample, target, path

    def Animal(split, transform, root):
        dataset = Animal(root=root, split=split, transform=transform)
        return dataset


class AnimalFeatureDataset(Animal):
    """Dataset of Animal90's pre-computed features."""

    def __init__(self, split, model, arch, root, transform=None, target_transform=None) -> Dataset:
        super().__init__(root=DATA_DIR['animal90'], split=split)
        self.featfile = os.path.join(root, model.lower(), arch.replace('/', '-'), f'animal90_{split}_features.pkl')
        assert os.path.isfile(self.featfile), f'featrue file {self.featfile} not found'
        with open(self.featfile, 'rb') as file:
            self.feature_data = pickle.load(file)
        self.feature_samples = self.feature_data['features']
        self.feature_dtype = self.feature_samples[0][2].dtype
        assert len(self.feature_samples) == len(
            self.samples
        ), f'number of features {len(self.feature_samples)} does not match number of images {len(self.samples)}'
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index) -> Tuple[Any, Any, str]:
        imgpath, _target = self.samples[index]
        imgname, target, feature = self.feature_samples[index]
        assert imgname in imgpath and _target == target

        if self.transform is not None:
            feature = self.transform(feature)
        if self.target_transform is not None:
            target = self.target_transform([target])

        return feature, target, imgpath
