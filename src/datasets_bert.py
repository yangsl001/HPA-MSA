# src/datasets_bert.py

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import numpy as np
from transformers import BertTokenizer


class MultimodalDataset_BERT(Dataset):
    """
    A PyTorch Dataset for loading multimodal data, specifically designed to handle
    raw text and tokenize it on-the-fly for BERT fine-tuning.
    """

    def __init__(self, data_partition, mode, config):
        """
        Args:
            data_partition (dict): The dictionary for a specific partition ('train', 'valid', or 'test').
            mode (str): The partition mode ('train', 'valid', or 'test').
            config (EasyDict): The main configuration object.
        """
        self.mode = mode
        self.config = config

        # 1. Initialize BERT Tokenizer
        print(f"Initializing BERT tokenizer for '{config.bert_model_name}'...")
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_name, do_lower_case=True)

        # 2. Load non-text features
        self.vision = torch.tensor(data_partition['vision'], dtype=torch.float32)
        self.audio = torch.tensor(data_partition['audio'], dtype=torch.float32)

        # 3. Load raw text
        if 'raw_text' not in data_partition:
            raise KeyError("The provided data partition does not contain the required 'raw_text' field.")
        self.raw_text = data_partition['raw_text']

        # 4. Load and process labels
        self.regression_labels = torch.tensor(data_partition['regression_labels'], dtype=torch.float32)

        # Apply label mapping for classification
        original_cls_labels = np.array(data_partition['classification_labels'])
        if config.num_classes == 2:
            # Rule: > 0 is positive (class 1), <= 0 is non-positive (class 0)
            corrected_cls_labels = (original_cls_labels > 0).astype(int)
        else:
            # Assumes labels are already in the correct range for multiclass
            corrected_cls_labels = original_cls_labels
        self.classification_labels = torch.tensor(corrected_cls_labels, dtype=torch.long)

        self.num_samples = len(self.raw_text)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        Fetches a single sample and tokenizes the raw text.
        """
        text_sentence = self.raw_text[index]

        # Tokenize the text sentence using the BERT tokenizer
        encoded_text = self.tokenizer.encode_plus(
            text_sentence,
            add_special_tokens=True,
            max_length=self.config.max_text_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'  # Return PyTorch tensors
        )

        sample = {
            # Squeeze the batch dimension (1) added by return_tensors='pt'
            'input_ids': encoded_text['input_ids'].squeeze(0),
            'attention_mask': encoded_text['attention_mask'].squeeze(0),
            'token_type_ids': encoded_text['token_type_ids'].squeeze(0),

            'vision': self.vision[index],
            'audio': self.audio[index],

            'labels': {
                'regression': self.regression_labels[index],
                'classification': self.classification_labels[index]
            }
        }
        return sample


def get_data_loaders_bert(config):
    """
    A helper function to load data from a .pkl file and create PyTorch DataLoaders
    for the BERT-integrated workflow.

    Args:
        config (EasyDict): The main configuration object.

    Returns:
        tuple: A tuple containing (train_loader, valid_loader, test_loader).
    """
    data_path = config.data_path
    print(f"Loading data for BERT fine-tuning from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)

    # Standardize partition keys
    partition_keys = {'train': 'train', 'valid': 'valid', 'test': 'test'}
    if 'valid' not in all_data and 'dev' in all_data:
        partition_keys['valid'] = 'dev'
        print("Found 'dev' partition, using it for validation.")

    # Create Dataset instances
    # Pass the full config object as it contains tokenizer and label mapping info
    train_dataset = MultimodalDataset_BERT(all_data[partition_keys['train']], 'train', config)
    valid_dataset = MultimodalDataset_BERT(all_data[partition_keys['valid']], 'valid', config)
    test_dataset = MultimodalDataset_BERT(all_data[partition_keys['test']], 'test', config)

    print(
        f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}, Test samples: {len(test_dataset)}")

    # Create DataLoader instances
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, valid_loader, test_loader