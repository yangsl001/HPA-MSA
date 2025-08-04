# src/datasets.py

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import numpy as np

class MultimodalDataset(Dataset):
    """
    A PyTorch Dataset for loading pre-processed multimodal data.
    The data is expected to be in a dictionary format, loaded from a .pkl file,
    where keys are 'train', 'valid' (or 'dev'), 'test', and values are dictionaries
    containing batched features and labels.
    """

    def __init__(self, data_partition, mode, num_classes):
        """
        Args:
            data_partition (dict): The dictionary for a specific partition
                                   (e.g., all_data['train']).
            mode (str): The partition mode, e.g., 'train', 'valid', or 'test'.
        """
        self.mode = mode

        # 1. Load feature tensors
        # Convert numpy arrays to PyTorch tensors.
        # The raw data might be float64, but models typically use float32 for efficiency.
        self.text = torch.tensor(data_partition['text'], dtype=torch.float32)
        self.vision = torch.tensor(data_partition['vision'], dtype=torch.float32)
        self.audio = torch.tensor(data_partition['audio'], dtype=torch.float32)

        # 2. Load label tensors
        self.regression_labels = torch.tensor(data_partition['regression_labels'], dtype=torch.float32)
        # Classification labels should be long integers for CrossEntropyLoss
        original_cls_labels = np.array(data_partition['classification_labels'])

        if num_classes == 2:
            corrected_cls_labels = (original_cls_labels == 2).astype(int)
        else:
            corrected_cls_labels = original_cls_labels

        self.classification_labels = torch.tensor(corrected_cls_labels, dtype=torch.long)
        # 3. Store the number of samples
        self.num_samples = self.text.shape[0]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, index):
        """
        Returns a single sample from the dataset at the given index.
        """
        sample = {
            'text': self.text[index],
            'vision': self.vision[index],
            'audio': self.audio[index],
            'labels': {
                'regression': self.regression_labels[index],
                'classification': self.classification_labels[index]
            }
        }
        return sample


def get_data_loaders(data_path, batch_size=32, num_workers=4, num_classes=2):
    """
    A helper function to load data from a .pkl file and create PyTorch DataLoaders.

    Args:
        data_path (str): Path to the .pkl file.
        batch_size (int): The batch size for the DataLoaders.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: A tuple containing (train_loader, valid_loader, test_loader).
    """
    print(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)

    partition_keys = {'train': 'train', 'valid': 'valid', 'test': 'test'}
    if 'valid' not in all_data and 'dev' in all_data:
        partition_keys['valid'] = 'dev'
        print("Found 'dev' partition, using it for validation.")

    try:
        # --- MODIFICATION: Pass num_classes to the Dataset constructor ---
        train_dataset = MultimodalDataset(all_data[partition_keys['train']], 'train', num_classes)
        valid_dataset = MultimodalDataset(all_data[partition_keys['valid']], 'valid', num_classes)
        test_dataset = MultimodalDataset(all_data[partition_keys['test']], 'test', num_classes)
    except KeyError as e:
        raise KeyError(f"The .pkl file must contain '{e}' key for data partitions.") from e

    print(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}, Test samples: {len(test_dataset)}")

    # Create DataLoader instances
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Speeds up data transfer to GPU
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation/test data
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, valid_loader, test_loader