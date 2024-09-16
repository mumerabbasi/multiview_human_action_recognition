import torch
import torch.nn as nn


class TransformerTemporal(nn.Module):
    """
    Transformer Encoder for temporal modeling of the sequence.

    Args:
        embed_dim (int): Embedding dimension (input feature size).
        num_heads (int): Number of attention heads.
        num_layers (int): Number of Transformer layers.
        dropout (float): Dropout rate.
    """
    def __init__(self, embed_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerTemporal, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                dropout=dropout
            ),
            num_layers=num_layers
        )

        # 50 is max sequence length
        self.positional_encoding = nn.Parameter(torch.randn(1, 50, embed_dim))

    def forward(self, x):
        """
        Forward pass through the Transformer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, embed_dim].

        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, embed_dim].
        """
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Apply Transformer Encoder
        return self.transformer_encoder(x)
