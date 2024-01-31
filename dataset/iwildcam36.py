import os
import torch
import pickle
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from typing import Callable, Tuple, Any, Optional

from . import DATA_DIR


class IWildCam(ImageFolder):
    dataset_name = 'iwildcam36'

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(os.path.join(root, split), transform, target_transform)
        self.split = split
        self.n_classes = len(self.classes)

    def __getitem__(self, index) -> Tuple[Any, Any, Any]:
        sample, target = super().__getitem__(index=index)
        path, _ = self.samples[index]
        return sample, target, path


class IWildCamFeatureDataset(IWildCam):
    """Dataset of ImageNet-1k's CLIP features, modified from `torchvision.datasets.ImageNet`."""

    def __init__(self, split, model, arch, root) -> Dataset:
        super().__init__(root=DATA_DIR[IWildCam.dataset_name], split=split)
        self.featfile = os.path.join(root, model.lower(), arch.replace('/', '-'), f'{self.dataset_name}_{split}_features.pkl')
        assert os.path.isfile(self.featfile), f'featrue file {self.featfile} not found'
        with open(self.featfile, 'rb') as file:
            self.feature_data = pickle.load(file)
        self.feature_samples = self.feature_data['features']
        self.feature_dtype = self.feature_samples[0][2].dtype
        assert len(self.feature_samples) == len(
            self.samples
        ), f'number of features {len(self.feature_samples)} does not match number of images {len(self.samples)}'

    def __getitem__(self, index) -> Tuple[Any, Any, str]:
        imgpath, _target = self.samples[index]
        imgname, target, feature = self.feature_samples[index]
        assert imgname in imgpath and _target == target
        return torch.HalfTensor(feature), torch.as_tensor(target), imgpath
