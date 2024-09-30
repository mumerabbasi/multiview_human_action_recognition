import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


def organize_actions(src_dir, target_dir, action_labels_csv, stride=24,
                     max_frame_diff=48, num_views=8):
    '''
    Organize actions into folders like /action_x/seq_x/view_x by reading
    action_labels.csv and building 8 action sequences (CSV has annotations
    for only one view) for each row.

    Args:
        src_dir (str): The source directory containing the raw data.
        target_dir (str): The directory where organized action folders will be
            stored.
        action_labels_csv (str): Path to the CSV file containing action labels.
        stride (int): Number of frames to stride between consecutive subgroups.
        max_frame_diff (int): Maximum number of frames to consider in a
            subgroup.
        num_views (int): The number of views (camera angles) per action.
    '''

    # Read the CSV file containing action labels and frame information.
    df = pd.read_csv(action_labels_csv)

    # Dictionary to keep track of the number of sequences for each action
    # label.
    action_seq_count = {}

    # Loop through each row in the CSV file (each row corresponds to an action
    # instance).
    for index, row in df.iterrows():
        # Get the action label and replace spaces with underscores.
        action_label = row['action_label'].replace(" ", "_")

        # Initialize the action sequence count if not already in the
        # dictionary.
        if action_label not in action_seq_count:
            action_seq_count[action_label] = 0

        # Get the total number of frames and start frame for this action.
        total_frames = row['frame_diff']
        start_frame = row['start frame']

        # Calculate the number of subgroups based on the stride value.
        num_subgroups = (total_frames - stride) // stride + 1

        # Limit total frames to the maximum allowed frame difference.
        total_frames = min(total_frames, max_frame_diff)

        # Loop through each subgroup and create sequences for each.
        for subgroup_index in range(num_subgroups):
            # Increment the action sequence count for this action label.
            action_seq_count[action_label] += 1
            seq_folder = f"seq_{action_seq_count[action_label]}"

            # Create folders for each view (camera angle).
            for view in range(1, num_views + 1):
                act_folder = os.path.join(
                    target_dir, action_label, seq_folder, f"view_{view}"
                )

                # Calculate the start and end frame for this subgroup.
                subgroup_start = start_frame + subgroup_index * stride
                subgroup_end = subgroup_start + max_frame_diff

                # Ensure the subgroup end does not exceed the total frames.
                subgroup_end = min(
                    subgroup_end, start_frame + row['frame_diff']
                )

                # Skip if the subgroup has fewer frames than the stride.
                if (subgroup_end - subgroup_start) < stride:
                    continue

                # Create the directory if it does not exist.
                if not os.path.exists(act_folder):
                    os.makedirs(act_folder)

                # Define the source folder path for the current view.
                src_folder = os.path.join(
                    src_dir,
                    row['sequence_number'],
                    f"{row['sequence_number']}_view_{view}"
                )

                # Copy each frame in the subgroup from the source to the
                # target folder.
                for frame in range(subgroup_start, subgroup_end + 1):
                    for ext in ['jpg']:
                        src_file = os.path.join(
                            src_folder, f"left{frame:04d}.{ext}"
                        )
                        tgt_file = os.path.join(
                            act_folder, f"left{frame:04d}.{ext}"
                        )

                        # Check if the source file exists and copy it,
                        # otherwise print a warning.
                        if os.path.exists(src_file):
                            shutil.copy(src_file, tgt_file)
                        else:
                            print(f"File does not exist: {src_file}")

        if num_subgroups == 0:
            print(
                f"Skipping action due to insufficient frames for row {index}"
            )

    print("Organized actions into separate folders.")


def stratified_split_dataset(
        data_dir, out_dir, train_pct=0.8, val_pct=0.1, test_pct=0.1
        ):

    """
    Stratified split of multiview dataset into train, validation, and test
    sets, while preserving the original directory structure and removing empty
    directories.

    Args:
        data_dir (str): Path to the directory containing the dataset. The
            structure should be like 'data/action_label/seq_x/view_x'.
        out_dir (str): Path to the output directory where train, val, and test
            directories will be created.
        train_pct (float): Percentage of data to allocate to the training set.
            Must be between 0 and 1.
        val_pct (float): Percentage of data to allocate to the validation set.
            Must be between 0 and 1.
        test_pct (float): Percentage of data to allocate to the test set.
            Must be between 0 and 1.

    Raises:
        AssertionError: If the sum of train_pct, val_pct, and test_pct is not
            equal to 1.

    The directory structure after splitting will look like:
        out_dir/train/action_label/seq_x/view_x
        out_dir/val/action_label/seq_x/view_x
        out_dir/test/action_label/seq_x/view_x
    """
    assert train_pct + val_pct + test_pct == 1.0, "Train, Val, and Test \
    percentages must sum to 1."

    # Get all action labels and sequences
    actions = [action for action in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, action))]
    action_to_seqs = {}

    for action in actions:
        action_path = os.path.join(data_dir, action)
        seqs = [seq for seq in os.listdir(action_path)
                if os.path.isdir(os.path.join(action_path, seq))]
        action_to_seqs[action] = seqs

    train_dir = os.path.join(out_dir, 'train')
    val_dir = os.path.join(out_dir, 'val')
    test_dir = os.path.join(out_dir, 'test')

    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # For each action, perform a stratified split on sequences
    for action, seqs in action_to_seqs.items():
        train_seqs, temp_seqs = train_test_split(
            seqs, test_size=(val_pct + test_pct), random_state=42
            )
        val_seqs, test_seqs = train_test_split(
            temp_seqs, test_size=(test_pct / (val_pct + test_pct)),
            random_state=42
            )

        # Move sequences to respective directories
        for seq in train_seqs:
            move_sequence(data_dir, train_dir, action, seq)
        for seq in val_seqs:
            move_sequence(data_dir, val_dir, action, seq)
        for seq in test_seqs:
            move_sequence(data_dir, test_dir, action, seq)

    # Remove empty directories after moving
    remove_empty_dirs(data_dir)


def move_sequence(data_dir, target_dir, action, seq):
    """
    Move a sequence directory from the data directory to the target directory,
    preserving the action and sequence structure.

    Args:
        data_dir (str): Path to the original data directory.
        target_dir (str): Path to the target directory where data should be
            moved.
        action (str): Action label (sub-directory name).
        seq (str): Sequence identifier (sub-directory name).
    """
    action_dir = os.path.join(target_dir, action)
    os.makedirs(action_dir, exist_ok=True)

    seq_dir = os.path.join(data_dir, action, seq)
    target_seq_dir = os.path.join(action_dir, seq)

    # Move the entire sequence directory
    shutil.move(seq_dir, target_seq_dir)


def remove_empty_dirs(root_dir):
    """
    Recursively remove empty directories from a root directory.

    Args:
        root_dir (str): Path to the root directory to clean up.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if not dirnames and not filenames:
            os.rmdir(dirpath)
