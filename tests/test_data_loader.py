import os
import unittest
from torch.utils.data import DataLoader
from src.data.data_loader import create_dataloaders


class TestCreateDataloaders(unittest.TestCase):
    def setUp(self):
        """
        Set up necessary variables for the tests.
        """
        self.data_dir = "data/processed"
        self.batch_size = 1
        self.num_views = 8
        self.seq_len = 25
        self.num_workers = 2
        self.transform = None

        # Ensure data directories exist for test
        assert os.path.exists(self.data_dir), \
            f"Data directory {self.data_dir} does not exist"
        assert os.path.exists(f"{self.data_dir}/train"), \
            "Train directory does not exist"
        assert os.path.exists(f"{self.data_dir}/val"), \
            "Validation directory does not exist"
        assert os.path.exists(f"{self.data_dir}/test"), \
            "Test directory does not exist"

    def test_dataloaders_creation(self):
        """
        Test if dataloaders are correctly created and are instances of
            DataLoader.
        """
        train_loader, val_loader, test_loader = create_dataloaders(
            self.data_dir, batch_size=self.batch_size, seq_len=self.seq_len,
            num_workers=self.num_workers, transform=self.transform
        )

        # Check if the returned objects are DataLoader instances
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)

    def test_dataloaders_not_empty(self):
        """
        Test if the created dataloaders are not empty.
        """
        train_loader, val_loader, test_loader = create_dataloaders(
            self.data_dir, batch_size=self.batch_size, seq_len=self.seq_len,
            num_workers=self.num_workers, transform=self.transform
        )

        # Check that dataloaders are not empty
        self.assertGreater(
            len(train_loader), 0, "Train dataloader is empty"
            )
        self.assertGreater(
            len(val_loader), 0, "Validation dataloader is empty"
            )
        self.assertGreater(
            len(test_loader), 0, "Test dataloader is empty"
            )

    def test_batch_size(self):
        """
        Test if the dataloaders return batches of the correct size.
        """
        train_loader, _, _ = create_dataloaders(
            self.data_dir, batch_size=self.batch_size, seq_len=self.seq_len,
            num_workers=self.num_workers, transform=self.transform
        )

        # Fetch a single batch from the train loader
        for batch in train_loader:
            batch_data, batch_labels = batch
            # Check the size of the first dimension (batch size)
            self.assertEqual(
                len(batch_data), self.batch_size, "Batch size mismatch"
                )
            break

    def test_num_view_in_dataset(self):
        """
        Test if the sequences in the dataset have the correct num views
        """
        _, val_loader, _ = create_dataloaders(
            self.data_dir, batch_size=self.batch_size, seq_len=self.seq_len,
            num_workers=self.num_workers, transform=self.transform
        )

        for batch in val_loader:
            batch_data, batch_labels = batch
            # Check sequence length in the dataset
            self.assertEqual(
                batch_data.size(1), self.num_views, "Num views mismatch"
                )
            break

    def test_seq_len_in_dataset(self):
        """
        Test if the sequences in the dataset have the correct length
            (if specified).
        """
        _, _, test_loader = create_dataloaders(
            self.data_dir, batch_size=self.batch_size, seq_len=self.seq_len,
            num_workers=self.num_workers, transform=self.transform
        )

        for batch in test_loader:
            batch_data, batch_labels = batch
            # Check sequence length in the dataset
            self.assertEqual(
                batch_data.size(2), self.seq_len, "Sequence length mismatch"
                )
            break


if __name__ == '__main__':
    unittest.main()
