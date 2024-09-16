import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class SpatialFeatureExtractor(nn.Module):
    """
    Spatial Feature Extractor block with EfficientNetV2-S.

    Args:
        num_views (int): Number of views in the input (e.g., 8 views).
        num_heads (int): Number of attention heads.
        pretrained (bool): Whether to use a pretrained EfficientNetV2-S model.
    """
    def __init__(self, pretrained=True):
        super(SpatialFeatureExtractor, self).__init__()

        # Load EfficientNetV2-S pre-trained on ImageNet
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.efficient_net = efficientnet_v2_s(weights=weights)

        # Remove the classification head (we only need the feature extractor)
        self.feature_extractor = nn.Sequential(*list(
            self.efficient_net.children())[:-1]
            )  # Removes the last FC layer

        # Freeze all layers of the feature extractor except the last conv
        #   block to finetune it
        # Freeze all layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # Unfreeze the last conv block
        for param in self.feature_extractor[-1].parameters():
            param.requires_grad = True

        # Dynamically determine embedding dimension (output feature dimension)
        #   using a dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            dummy_output = self.feature_extractor(dummy_input)
        self.embed_dim = dummy_output.shape[1]

    def forward(self, x):
        """
        Forward pass through the spatial feature extractor.

        Args:
            x (Tensor): Input tensor of shape [batch_size, num_views, C, H, W].

        Returns:
            Tensor: Extracted features with attention applied,
                    shape [batch_size, num_views, embed_dim].
        """
        batch_size, num_views, seq_len, C, H, W = x.shape

        # Combine batch_size and num_views into a single dimension
        x = x.view(batch_size * num_views * seq_len, C, H, W)

        # Pass the combined tensor through the feature extractor
        # Shape [batch_size * num_views * seq_len, embed_dim, 1, 1] after
        #   EfficientNetV2-S
        features = self.feature_extractor(x)
        # Shape [batch_size * num_views, embed_dim]
        features = features.squeeze(-1).squeeze(-1)

        # Reshape the features back to [batch_size, num_views, embed_dim]
        # Shape [batch_size, num_views, embed_dim]
        features = features.view(batch_size, num_views, seq_len, -1)

        return features  # Shape [batch_size, num_views, seq_len, embed_dim]
