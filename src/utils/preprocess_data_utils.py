import os
import shutil
import pandas as pd


def organize_actions(src_dir, target_dir, action_labels_csv, stride=24,
                     max_frame_diff=48, num_views=8):
    '''
    Organize actions into folders like /action_x/seq_x/view_x by reading
    action_labels.csv and building 8 action sequences (CSV has annotations
    for only one view) for each row.

    Args:
        src_dir: The source directory containing the raw data.
        target_dir: The directory where organized action folders will be
            stored.
        action_labels_csv: Path to the CSV file containing action labels.
        stride: Number of frames to stride between consecutive subgroups.
        max_frame_diff: Maximum number of frames to consider in a subgroup.
        num_views: The number of views (camera angles) per action.
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
                    for ext in ['jpg']:  # Assuming files are in .jpg format
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

        # Print a warning if there are not enough frames to create any
        # subgroups.
        if num_subgroups == 0:
            print(
                f"Skipping action due to insufficient frames for row {index}"
            )

    # Print a message once the organization process is complete.
    print("Organized actions into separate folders.")
