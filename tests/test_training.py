import torch
import unittest
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from src.models.multiview_action_recognition_model import (
    MultiviewActionRecognitionModel
)
from src.data.dataset import MultiviewActionDataset


class TestOverfitSingleSequence(unittest.TestCase):
    def setUp(self):
        """
        This method is called once before all the tests. It initializes the
            model, dataloader, loss function, and optimizer. It also loads a
            single sequence for testing overfitting.
        """
        # Hyperparameters
        self.batch_size = 1
        self.num_epochs = 100
        self.learning_rate = 1e-4

        # Device configuration (GPU or CPU)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
            )

        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize frames to a standard size
            transforms.ToTensor(),  # Convert images to tensors
        ])

        # Dataset and Dataloader (load only one sequence for overfitting)
        train_dataset = MultiviewActionDataset(
            data_dir='data/processed/train', transform=transform
        )

        # Get number of classes form the dataset
        self.num_classes = train_dataset.num_classes

        # Load a single sequence for overfitting test
        single_sequence_data = [train_dataset[0]]
        self.single_sequence_loader = DataLoader(
            single_sequence_data, batch_size=self.batch_size, shuffle=False
            )

        # Initialize model, loss function, optimizer
        self.model = MultiviewActionRecognitionModel(
            num_heads=4, num_transformer_layers=2, num_classes=self.num_classes
            )
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()  # Loss function
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate
            )  # Adam optimizer

    def test_overfit_on_one_sequence(self):
        """
        This test checks if the model can overfit a single sequence.
        """
        print("Starting Overfitting Test on One Sequence...")
        for epoch in range(self.num_epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            # Since we are testing overfitting, we only have one sequence to
            #   train on
            for batch_idx, (views, actions) in enumerate(
                self.single_sequence_loader
            ):
                views, actions = views.to(self.device), actions.to(self.device)

                # Forward pass
                outputs = self.model(views)
                loss = self.criterion(outputs, actions)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Calculate statistics
                _, predicted = torch.max(outputs, 1)
                total_predictions += actions.size(0)
                correct_predictions += (predicted == actions).sum().item()
                running_loss += loss.item()

            # Print loss and accuracy for this epoch
            epoch_loss = running_loss / total_predictions
            epoch_accuracy = 100 * correct_predictions / total_predictions

            print(
                    f"Epoch [{epoch+1}/{self.num_epochs}], "
                    f"Loss: {epoch_loss:.4f}, "
                    f"Accuracy: {epoch_accuracy:.2f}%"
                )

            # Stop training if the model has perfectly overfitted the sequence
            if correct_predictions == total_predictions:
                print(
                    f"Model successfully overfitted on the sequence "
                    f"at Epoch {epoch+1}."
                )
                break

        # Assert that the model has overfitted on the sequence
        self.assertEqual(correct_predictions, total_predictions,
                         "Model failed to overfit on a single sequence")


if __name__ == "__main__":
    unittest.main()
