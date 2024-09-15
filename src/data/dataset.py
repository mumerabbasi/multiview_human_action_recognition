import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob


class MultiviewActionDataset(Dataset):
    """
    Custom Dataset for Multiview Action Recognition.

    Args:
        data_dir (str): Path to the processed dataset directory
        (e.g., 'data/processed').
        transform (callable, optional): A function/transform that takes in an
        image and returns a transformed version.
        seq_len (int, optional): Fixed length for sequences. Default is None
        (uses original length).
    """
    def __init__(self, data_dir, transform=None, seq_len=None):
        self.data_dir = data_dir
        self.transform = transform
        self.seq_len = seq_len
        self.actions = sorted(os.listdir(data_dir))
        self.data = self._load_dataset()

    def _load_dataset(self):
        """
        Internal method to load the dataset paths into memory.
        Collects file paths of all frames for each sequence from each view.

        Returns:
            List of tuples in the format (action, sequence_id, view_paths)
        """
        data = []
        for action in self.actions:
            action_dir = os.path.join(self.data_dir, action)
            sequences = sorted(os.listdir(action_dir))
            for seq in sequences:
                seq_dir = os.path.join(action_dir, seq)
                views = sorted(glob.glob(os.path.join(seq_dir, '*')))
                data.append((action, seq, views))
        return data

    def __len__(self):
        return len(self.data)

    def _load_images(self, view_paths):
        """
        Internal method to load images for all views of a single sequence.

        Args:
            view_paths (list): List of paths for each view in the sequence.

        Returns:
            List of tensors for each view (one tensor per view).
        """
        images_per_view = []
        for view_path in view_paths:
            frames = sorted(glob.glob(os.path.join(view_path, '*.jpg')))
            if self.seq_len:
                frames = frames[:self.seq_len]
            images = [Image.open(frame) for frame in frames]
            if self.transform:
                images = [self.transform(image) for image in images]
            else:
                images = [transforms.ToTensor()(image) for image in images]
            # Shape: [seq_len, C, H, W]
            images_per_view.append(torch.stack(images))

        # Stack views into a single tensor, adding a view dimension
        # Shape: [num_views, seq_len, C, H, W]
        return torch.stack(images_per_view, dim=0)

    def __getitem__(self, idx):
        """
        Fetches the data at the specified index.

        Args:
            idx (int): Index for the item to fetch.

        Returns:
            dict: Contains action label, sequence ID, and view frames.
        """
        action, seq, view_paths = self.data[idx]
        views = self._load_images(view_paths)

        return {
            'action': action,
            'views': views  # List of [seq_len, C, H, W] for each view
        }
