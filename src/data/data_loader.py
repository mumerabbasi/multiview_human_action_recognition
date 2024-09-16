from torch.utils.data import DataLoader
from src.data.dataset import MultiviewActionDataset


def create_dataloaders(
        data_dir, batch_size=8, seq_len=None, num_workers=4, transform=None
        ):
    """
    Function to create PyTorch dataloaders for training, validation, and
    testing.

    Args:
        data_dir (str): Path to the processed dataset directory.
        batch_size (int): Batch size for the dataloader.
        seq_len (int, optional): Fixed length for sequences. If None, original
            sequence length is used.
        num_workers (int): Number of workers for data loading.
        transform (callable, optional): A function/transform to apply to the
            images.

    Returns:
        train_loader, val_loader, test_loader: Dataloaders for training,
            validation, and testing sets.
    """
    # Define dataset splits (adjust these as necessary)
    train_dir = f"{data_dir}/train"
    val_dir = f"{data_dir}/val"
    test_dir = f"{data_dir}/test"

    # Create dataset instances
    train_dataset = MultiviewActionDataset(
        train_dir, transform=transform, seq_len=seq_len
        )
    val_dataset = MultiviewActionDataset(
        val_dir, transform=transform, seq_len=seq_len
        )
    test_dataset = MultiviewActionDataset(
        test_dir, transform=transform, seq_len=seq_len
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers
        )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
        )
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers
                             )

    return train_loader, val_loader, test_loader
