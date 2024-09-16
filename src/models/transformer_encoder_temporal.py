import torch.nn as nn


class TemporalTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        """
        Transformer block for modeling temporal/sequence length dimension.

        Args:
            embed_dim (int): Dimensionality of the input embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            dim_feedforward (int): Dimensionality of the feedforward network
                in the transformer layers.
            dropout (float): Dropout probability.
        """
        super(TemporalTransformerBlock, self).__init__()

        # Transformer encoder layer that will operate on the temporal/sequence
        #   length dimension
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,  # The feature dimension (embed_dim)
            nhead=num_heads,    # Number of attention heads
            dim_feedforward=2*embed_dim,  # Size of the feedforward network
            dropout=dropout,    # Dropout rate
            batch_first=True     # Ensures batch is the first dimension
        )

        # Stack multiple transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
            )

    def forward(self, x):
        """
        Forward pass through the temporal transformer block.

        Args:
            x (Tensor): Input tensor of shape
                [batch_size, num_views, seq_len, embed_dim].

        Returns:
            Tensor: Output tensor of shape
                [batch_size, num_views, seq_len, embed_dim].
        """
        batch_size, num_views, seq_len, embed_dim = x.shape

        # We need to model the sequence/temporal dimension (seq_len)
        # Reshape the input to combine batch_size and num_views for processing
        #   the temporal dimension
        # Shape: [batch_size * num_views, seq_len, embed_dim]
        x = x.view(batch_size * num_views, seq_len, embed_dim)

        # Apply transformer encoder on the temporal dimension (seq_len)
        # Shape remains [batch_size * num_views, seq_len, embed_dim]
        x = self.transformer_encoder(x)

        # Reshape back to [batch_size, num_views, seq_len, embed_dim]
        x = x.view(batch_size, num_views, seq_len, embed_dim)

        return x
