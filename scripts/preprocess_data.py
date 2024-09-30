import os
from src.utils.preprocess_data_utils import (
    organize_actions, stratified_split_dataset)
from src.utils.helpers import load_config


def main():

    # Define configs path and load configs
    default_config_path = os.path.join('configs', 'default.yaml')
    custom_config_path = os.path.join('configs', 'custom.yaml')

    # Load the combined configuration
    config = load_config(default_config_path, custom_config_path)

    # Define data paths
    raw_data_path = os.path.join('data', 'raw')
    processed_data_path = os.path.join('data', 'processed')
    action_labels_csv = os.path.join(
        'data', 'annotations', 'action_labels.csv')

    # Organizes the data into /action_x/seq_x/view_x folders by reading
    # action_labels.csv
    print("Organizing actions into folders...")
    organize_actions(
        raw_data_path, processed_data_path, action_labels_csv
        )

    # Access the train, val, and test percentages
    train_pct = config.get('train_pct')
    val_pct = config.get('val_pct')
    test_pct = config.get('test_pct')

    print("Splitting data into train, validation and test sets...")
    stratified_split_dataset(
        processed_data_path, processed_data_path, train_pct, val_pct, test_pct
        )
    print("Preprocessing completed.")


if __name__ == "__main__":
    main()
