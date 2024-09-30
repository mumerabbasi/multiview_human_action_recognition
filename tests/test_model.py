import os
import unittest
from src.data.data_loader import create_dataloaders
from src.models.attention_views import MultiHeadAttentionOnViews
from src.models.spatial_feature_extractor import SpatialFeatureExtractor
from src.models.transformer_encoder_temporal import TemporalTransformerBlock
from src.models.multiview_action_recognition_model import (
    MultiviewActionRecognitionModel
)


class TestMultiviewActionRecognitionModel(unittest.TestCase):

    def setUp(self):
        # Set up data loaders using the user's data loader script
        data_dir = "data/processed"
        batch_size = 1  # For overfitting test, we use batch size of 1
        num_workers = 2
        transform = None  # You can add any necessary transformations

        # Ensure data directories exist for the test
        assert os.path.exists(data_dir), \
            f"Data directory {data_dir} does not exist"
        assert os.path.exists(f"{data_dir}/train"), \
            f"Train data directory {data_dir}/train does not exist"

        # Create data loaders
        self.train_loader, _, _ = create_dataloaders(
            data_dir, batch_size=batch_size,
            num_workers=num_workers, transform=transform
        )

        # Initialize model parameters
        self.num_heads = 4
        self.num_transformer_layers = 2

        # Get number of actions/classes from data dir
        self.actions = sorted(os.listdir(data_dir))
        self.num_classes = len(self.actions)

        # Initialize the model
        self.model = MultiviewActionRecognitionModel(
            num_heads=self.num_heads,
            num_transformer_layers=self.num_transformer_layers,
            num_classes=self.num_classes
        )

    def test_model_forward_pass(self):
        """
        Test the forward pass of the model with actual data from the data
            loader.
        """
        # Get a single batch from the train loader
        for batch in self.train_loader:
            # images: [batch_size, num_views, seq_len, C, H, W]
            images, labels = batch
            output = self.model(images)
            self.assertIsNotNone(output, "Model output is None")
            # Check output shape
            self.assertEqual(output.shape[0], images.shape[0],
                             "Output batch size mismatch")
            self.assertEqual(output.shape[1], self.num_classes,
                             "Output class size mismatch")
            break  # Only need one batch for the test

    def test_components_with_real_data(self):
        """
        Test individual components with actual data from the data loader.
        """
        # Get a single batch from the train loader
        for batch in self.train_loader:
            images, labels = batch
            break

        # Initialize individual components
        spatial_feature_extractor = SpatialFeatureExtractor()
        multihead_attention_on_views = MultiHeadAttentionOnViews(
            embed_dim=spatial_feature_extractor.embed_dim,
            num_heads=self.num_heads
            )
        temporal_transformer_block = TemporalTransformerBlock(
            embed_dim=spatial_feature_extractor.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_transformer_layers
            )

        # Test spatial feature extractor
        spatial_features = spatial_feature_extractor(images)
        self.assertEqual(
            spatial_features.shape,
            (images.shape[0], images.shape[1], images.shape[2],
             spatial_feature_extractor.embed_dim),
            "Spatial feature extractor output shape mismatch"
        )

        # Test multi-head attention on views
        attention_output = multihead_attention_on_views(spatial_features)
        self.assertEqual(
            attention_output.shape,
            (images.shape[0], images.shape[1], images.shape[2],
             spatial_feature_extractor.embed_dim),
            "Multi-head attention output shape mismatch"
        )

        # Test temporal transformer block
        temporal_output = temporal_transformer_block(attention_output)
        self.assertEqual(
            temporal_output.shape,
            (images.shape[0], images.shape[1], images.shape[2],
             spatial_feature_extractor.embed_dim),
            "Temporal transformer block output shape mismatch"
        )


if __name__ == '__main__':
    unittest.main()
