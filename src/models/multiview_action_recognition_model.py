import torch.nn as nn
from src.models.spatial_feature_extractor import SpatialFeatureExtractor
from src.models.attention_views import MultiHeadAttentionOnViews
from src.models.transformer_encoder_temporal import TemporalTransformerBlock


class MultiviewActionRecognitionModel(nn.Module):
    """
    Multiview Action Recognition Model using EfficientNetV2, Multi-Head
    Attention on views, and Transformer Encoder for temporal modeling.

    Parameters:
        num_heads (int): Number of attention heads.
        pretrained_spatial_feature_extractor (bool): Whether to use a
            pretrained model
        num_transformer_layers (int): Number of Transformer layers.
        num_classes (int): Number of output classes.
    """
    def __init__(
            self, num_heads=4, pretrained_spatial_feature_extractor=True,
            num_transformer_layers=2, num_classes=10
            ):
        super(MultiviewActionRecognitionModel, self).__init__()

        # Feature extractor
        self.spatial_feature_extractor = SpatialFeatureExtractor(
            pretrained_spatial_feature_extractor
            )

        # Embedding dimension of the feature extractor
        self.embed_dim = self.spatial_feature_extractor.embed_dim

        # Multi-Head Attention on views
        self.attention_on_views = MultiHeadAttentionOnViews(
            self.embed_dim, num_heads
            )

        # Transformer for temporal modeling
        self.temporal_model = TemporalTransformerBlock(
            self.embed_dim, num_heads, num_transformer_layers
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the entire model.

        Args:
            x (Tensor): Input tensor of shape
            [batch_size, num_views, seq_len, C, H, W].

        Returns:
            Tensor: Output logits of shape [batch_size, num_classes].
        """
        # Extract spatial features for each frame
        # Shape: [batch_size, num_views, seq_len, embed_dim]
        features = self.spatial_feature_extractor(x)

        # Apply multi-head attention on views
        # Shape: [batch_size, num_views, seq_len, embed_dim]
        attn_output = self.attention_on_views(features)

        # Apply temporal Transformer on sequence
        # Shape: [batch_size, num_views, seq_len, embed_dim]
        temporal_transformer_output = self.temporal_model(attn_output)

        # Pooling features instead of concatenating them because the dataset
        # is small and concatenating features might result in overfitting

        # Pool over the temporal dimension (seq_len)
        # Shape: [batch_size, num_views, embed_dim]
        pooled_features = temporal_transformer_output.mean(dim=2)
        # Pool over the views dimension (num_views)
        # Shape: [batch_size, embed_dim]
        final_features = pooled_features.mean(dim=1)

        # Classification
        # Shape: [batch_size, num_classes]
        logits = self.classifier(final_features)

        return logits
