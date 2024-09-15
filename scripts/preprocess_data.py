import os
from src.utils.preprocess_data_utils import organize_actions


def main():
    # Define paths
    raw_data_path = os.path.join('data', 'raw')
    processed_data_path = os.path.join('data', 'processed')
    action_labels_csv = os.path.join(
        'data', 'annotations', 'action_labels.csv')

    # Organizes the data into /action_x/seq_x/view_x folders by reading
    # action_labels.csv
    print("Organizing actions into folders...")
    organize_actions(raw_data_path, processed_data_path, action_labels_csv)

    print("Preprocessing completed.")


if __name__ == "__main__":
    main()
