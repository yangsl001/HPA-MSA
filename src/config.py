# src/config.py

import yaml
import argparse
import torch
from easydict import EasyDict as edict
import os


def load_config(path):
    """
    Loads a YAML configuration file and resolves inheritance.

    Args:
        path (str): The path to the YAML configuration file.

    Returns:
        dict: The loaded configuration dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # Resolve inheritance: if a config file has an 'inherits' key,
    # load the base config first and then update it with the current one.
    if 'inherits' in config_dict:
        base_config_path = config_dict['inherits']

        # If the path is relative, it's relative to the directory of the current config file.
        if not os.path.isabs(base_config_path):
            current_dir = os.path.dirname(path)
            base_config_path = os.path.join(current_dir, base_config_path)

        base_config = load_config(base_config_path)
        # Update the base config with the specific settings from the child config
        base_config.update(config_dict)
        config_dict = base_config
        # The 'inherits' key is no longer needed
        del config_dict['inherits']

    return config_dict


def get_config():
    """
    Parses the command line for a config file path, loads the configuration,
    and returns it as an EasyDict object for easy access.

    Returns:
        EasyDict: A configuration object allowing dot notation access (e.g., config.batch_size).
    """
    # 1. Create a parser to get the single '--config' argument from the command line
    parser = argparse.ArgumentParser(
        description='Load a YAML configuration file for HPA-MSA model training.'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file (e.g., configs/cmu_mosi_default.yaml).'
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    # 2. Load the specified YAML file into a dictionary
    config_dict = load_config(args.config)

    # 3. Convert the dictionary to an EasyDict object.
    # This allows accessing dictionary keys as attributes (e.g., config.batch_size).
    config = edict(config_dict)

    # 4. Automatically set the device for training.
    # This avoids repetitive if/else checks in the training script.
    if config.get('use_gpu', False) and torch.cuda.is_available():
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')

    return config


if __name__ == '__main__':
    # This is a simple test to demonstrate how to use this script.
    # To run this test, you need to provide a config file path via the command line,
    # for example: python src/config.py --config configs/cmu_mosi_default.yaml

    print("--- Testing config loader ---")
    try:
        config = get_config()
        print("\nConfiguration loaded successfully:")
        # The edict object can be printed like a dictionary
        print(yaml.dump(config, default_flow_style=False))

        print("\nAccessing values using dot notation:")
        print(f"  Dataset: {config.dataset_name}")
        print(f"  Learning Rate: {config.learning_rate}")
        print(f"  Device: {config.device}")

    except SystemExit:
        print("\nPlease provide a config file path, e.g., --config configs/cmu_mosi_default.yaml")
    except Exception as e:
        print(f"\nAn error occurred: {e}")