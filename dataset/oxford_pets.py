import os
import PIL
import torch
import pickle
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from typing import Callable, Tuple, Any, Optional

from . import DATA_DIR


class OxfordPet(torchvision.datasets.OxfordIIITPet):
    dataset_name = 'aircraft'

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self.dataset_name = 'aircraft'
        super().__init__(
            root=DATA_DIR[self.dataset_name],
            split=split,
            annotation_level='variant',
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.layer_classes = [self.classes]
        self.class_to_idx = [self.class_to_idx]
        self.layer_targets = [torch.LongTensor(self._labels)]
        for level in ('family', 'manufacturer'):
            _split = torchvision.datasets.FGVCAircraft(
                root=DATA_DIR[self.dataset_name],
                split=split,
                annotation_level=level,
                transform=transform,
                target_transform=target_transform,
                download=download,
            )
            assert self._image_files == _split._image_files
            self.layer_classes = [_split.classes.copy()] + self.layer_classes
            self.layer_targets = [torch.LongTensor(_split._labels.copy())] + self.layer_targets
            del _split

        self.targets = self._labels
        self.layer_targets = torch.stack(self.layer_targets, dim=1)

        self.split = split
        self.n_classes = len(self.classes)

    def __getitem__(self, index) -> Tuple[Any, Any, Any]:
        path = self._image_files[index]
        target = self.targets[index]
        image = PIL.Image.open(path).convert("RGB")  # note that PIL was used to load images

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, target, path


class FGVCAircraftFeature(FGVCAircraft):
    """Dataset of FGVCAircraft's pre-computed features."""

    def __init__(self, split, model, arch, root, transform=None, target_transform=None) -> Dataset:
        super().__init__(root=DATA_DIR[FGVCAircraft.dataset_name], split=split)
        self.featfile = os.path.join(root, model.lower(), arch.replace('/', '-'), f'{self.dataset_name}_{split}_features.pkl')
        assert os.path.isfile(self.featfile), f'featrue file {self.featfile} not found'
        with open(self.featfile, 'rb') as file:
            self.feature_data = pickle.load(file)
        self.feature_samples = self.feature_data['features']
        self.feature_dtype = self.feature_samples[0][2].dtype
        assert len(self.feature_samples) == len(
            self
        ), f'number of features {len(self.feature_samples)} does not match number of images {len(self)}'
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index) -> Tuple[Any, Any, str]:
        imgpath = self._image_files[index]
        _target = self._labels[index]
        imgname, target, feature = self.feature_samples[index]
        assert imgname in imgpath and _target == target

        if self.transform is not None:
            feature = self.transform(feature)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return feature, target, imgpath
