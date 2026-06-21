# datasets.py: MNIST dataset classes with transform/target_transform support.

from .mnist import load_images, load_labels
from . import transforms as T


class MNISTDataset:
    def __init__(self, split, transform=None, target_transform=None, dataset_dir=None):
        images = load_images(split, dataset_dir=dataset_dir)
        labels = load_labels(split, dataset_dir=dataset_dir)
        self.images = transform(images) if transform is not None else images
        self.targets = target_transform(labels) if target_transform is not None else labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


class MulticlassDataset(MNISTDataset):
    def __init__(self, split, transform=None, target_transform=None, dataset_dir=None):
        if transform is None:
            transform = lambda x: T.to_flat(T.normalize(x))
        if target_transform is None:
            target_transform = T.one_hot
        super().__init__(split, transform, target_transform, dataset_dir=dataset_dir)


class BinaryDataset(MNISTDataset):
    def __init__(self, split, transform=None, target_transform=None, dataset_dir=None):
        if transform is None:
            transform = lambda x: T.to_flat(T.normalize(x))
        if target_transform is None:
            target_transform = T.binarize
        super().__init__(split, transform, target_transform, dataset_dir=dataset_dir)


class RegressionDataset(MNISTDataset):
    def __init__(self, split, transform=None, target_transform=None, dataset_dir=None):
        if transform is None:
            transform = lambda x: T.to_flat(T.normalize(x))
        if target_transform is None:
            target_transform = T.to_regression
        super().__init__(split, transform, target_transform, dataset_dir=dataset_dir)
