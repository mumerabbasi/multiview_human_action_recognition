import yaml
from pathlib import Path


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
