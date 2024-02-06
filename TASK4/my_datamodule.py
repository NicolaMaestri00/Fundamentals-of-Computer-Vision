# TUWIEN - WS2023 CV: Task4 - Mask Classification using CNN
# *********+++++++++*******++++INSERT GROUP NO. HERE
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import PIL
import numpy as np


def pil_loader(p):
    return PIL.Image.open(p)

class DataModule:

    def __init__(self, data_dir: str = 'facemask', img_size: int = 64, batch_size: int = 32, augmented=False, gray: bool = False, num_workers=1, preload=False):
        """
        Initializes the DataModule.
        
        Args:
        - data_dir (str): Path to the directory of the data.
        - img_size (int): Size of the images.
        - batch_size (int): Number of images used for each iteration.
        - augmented (bool): True if the data should be augmented.
        - num_workers (int): Number of worker threads that load the data.
        - gray (bool) : True if the data should be grayscaled.
        - preload (bool): True if the data should only be loaded once from disk.
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.augmented = augmented
        self.num_workers = num_workers
        self.gray = gray

        self.train_dataset = FaceMaskDataset(self.data_dir + '/train', transform=self.get_transforms(
            train=True), label_transform=self.label_transform(), preload=preload)
        self.val_dataset = FaceMaskDataset(self.data_dir + '/val', transform=self.get_transforms(
        ), label_transform=self.label_transform(), preload=preload)
        self.test_dataset = FaceMaskDataset(self.data_dir + '/test', transform=self.get_transforms(
        ), label_transform=self.label_transform(), preload=preload)

    def label_transform(self):
        return torch.Tensor

    def get_transforms(self, train: bool = False):
        """
        Returns transformations that should be applied to the dataset.

        Args:
        - train (bool): If true, training transformations are returned. If self.augmented and train is true, add data augmentation.
        
        Returns:
        - data_transforms: Transforms.Compose([...]), Transforms.ToTensor(), Transforms.Resize((...)), Transforms.RandomHorizontalFlip() and Transforms.RandomAffine(...).
        """
        data_transforms = None

        # student code start
        data_transforms = [
            transforms.ToTensor(),
            transforms.Resize((self.img_size, self.img_size)),
        ]
        if train and self.augmented:
            data_transforms += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=30),
                transforms.RandomErasing()
            ]

        data_transforms = transforms.Compose(data_transforms)
        # student code end

        return data_transforms

    def train_dataloader(self):
        """
        Returns the train dataloader.
        """
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Returns the value dataloader.
        """
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        """
        Returns the test dataloader.
        """
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False, num_workers=self.num_workers)


class FaceMaskDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, label_transform=None, preload=False):
        """
        Initializes the Face Mask Dataset.
        
        Args:
        - data_dir (str): Subdirectory of the facemask dataset.
        - transform: Transformations for the dataset.
        - label_transform: Transformations applied to the labels.
        - preload (bool): True if the data should be loaded only once.
        """
        self.preload = preload
        self.data_dir = data_dir
        self.face_paths = [
            f'{data_dir}/face/{name}' for name in os.listdir(f'{data_dir}/face')]
        self.mask_paths = [
            f'{data_dir}/mask/{name}' for name in os.listdir(f'{data_dir}/mask')]
        if preload:
            self.faces = [pil_loader(img_path)
                          for img_path in self.face_paths]
            self.masks = [pil_loader(img_path)
                          for img_path in self.mask_paths]
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.face_paths) + len(self.mask_paths)

    def __getitem__(self, idx: int):
        """
        Given an index, returns a sample of the dataset.
        
        Args:
        - idx (int): Index of the sample.
        """

        # take care of the label
        if idx < len(self.face_paths):
            label = 0
            if self.preload:
                image = self.faces[idx]
            else:
                img_path = self.face_paths[idx]
        else:
            label = 1
            if self.preload:
                image = self.masks[idx - len(self.face_paths)]
            else:
                img_path = self.mask_paths[idx - len(self.face_paths)]

        if not self.preload:
            image = pil_loader(img_path)

        if self.label_transform:
            label = self.label_transform([label])
        if self.transform:
            image = self.transform(image)
        return image, label.float()