import torch
import torch.nn as nn
from efficientnetv2 import EfficientNetV2FeatureExtractor
from attention_view import MultiHeadAttentionOnViews
from transformer_temporal import TransformerTemporal


class MultiviewActionRecognitionModel(nn.Module):
    """
    Multiview Action Recognition Model using EfficientNetV2, Multi-Head
    Attention on views, and Transformer Encoder for temporal modeling.

    Parameters:
        embed_dim (int): The embedding dimension of the input features.
        num_heads (int): Number of attention heads.
        num_transformer_layers (int): Number of Transformer layers.
        num_classes (int): Number of output classes.
    """
    def __init__(
            self, embed_dim=1280, num_heads=4,
            num_transformer_layers=2, num_classes=10
            ):
        super(MultiviewActionRecognitionModel, self).__init__()

        # Feature extractor
        self.feature_extractor = EfficientNetV2FeatureExtractor()

        # Multi-Head Attention on views
        self.attention_on_views = MultiHeadAttentionOnViews(
            embed_dim, num_heads
            )

        # Transformer for temporal modeling
        self.temporal_model = TransformerTemporal(
            embed_dim, num_heads, num_transformer_layers
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
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
        batch_size, num_views, seq_len, C, H, W = x.size()

        # Extract features for each view
        features = []
        for view in range(num_views):
            view_features = []
            for frame in range(seq_len):
                frame_features = self.feature_extractor(x[:, view, frame])
                view_features.append(frame_features)
            # Shape: [batch_size, seq_len, embed_dim]
            view_features = torch.stack(view_features, dim=1)
            features.append(view_features)

        # Stack features from all views:
        # [batch_size, num_views, seq_len, embed_dim]
        features = torch.stack(features, dim=1)

        # Apply multi-head attention on views
        # Shape: [batch_size, seq_len, embed_dim]
        attn_output = self.attention_on_views(features)

        # Apply temporal Transformer on sequence
        # Shape: [batch_size, seq_len, embed_dim]
        transformer_output = self.temporal_model(attn_output)

        # Global average pooling over the sequence
        # Shape: [batch_size, embed_dim]
        pooled_output = transformer_output.mean(dim=1)

        # Classification
        # Shape: [batch_size, num_classes]
        logits = self.classifier(pooled_output)

        return logits
