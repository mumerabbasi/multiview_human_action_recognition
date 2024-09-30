import yaml
from pathlib import Path
import torch

def load_config(default_config_path, custom_config_path=None):
    """
    Loads configuration from the default YAML file and optionally overrides 
        it with values from a custom YAML file.

    Args:
        default_config_path (str or Path): Path to the default YAML
            configuration file.
        custom_config_path (str or Path, optional): Path to the custom YAML
            file. If not provided, the default configuration is used as is.

    Returns:
        dict: The combined configuration, with custom values overriding
            defaults where applicable.
    """
    # Load default configuration
    with open(default_config_path, 'r') as default_file:
        config = yaml.safe_load(default_file)

    # If a custom config is provided, load it and update the default
    #   configuration
    if custom_config_path and Path(custom_config_path).exists():
        with open(custom_config_path, 'r') as custom_file:
            custom_config = yaml.safe_load(custom_file)
            if custom_config:
                # Overwrite defaults with custom values
                config.update(custom_config)

    return config


def save_checkpoint(filepath, model, optimizer, loss):
    """
    Save model checkpoint.

    Args:
        filepath (str): Path to save the checkpoint.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        loss (float): The validation loss at the time of saving.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)


def load_checkpoint(filepath, model, optimizer):
    """
    Load model checkpoint.

    Args:
        filepath (str): Path to load the checkpoint from.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.

    Returns:
        loss: The validation loss at the time of saving.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['loss']


def get_lr(optimizer):
    """
    Get learning rate from the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to get the learning rate from.

    Returns:
        float: Current learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
