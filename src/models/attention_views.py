import torch.nn as nn


class MultiHeadAttentionOnViews(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        """
        Multi-head attention block for views.

        Args:
            embed_dim (int): Dimensionality of the input embeddings.
            num_heads (int): Number of attention heads.
        """
        super(MultiHeadAttentionOnViews, self).__init__()

        # Multi-head attention layer from PyTorch
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            batch_first=True, dropout=0.1
            )

    def forward(self, x):
        """
        Forward pass through the multi-head attention on views.

        Args:
            x (Tensor): Input tensor of shape
                [batch_size, num_views, seq_len, embed_dim].

        Returns:
            Tensor: Output tensor with attention applied over views, shape
                [batch_size, num_views, seq_len, embed_dim].
        """
        batch_size, num_views, seq_len, embed_dim = x.shape

        # Reshape input for attention over views:
        # Combine batch_size and seq_len for multi-head attention over views:
        # Change shape to [batch_size, seq_len, num_views, embed_dim] as we
        #   need to apply attention on the views dim
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size * seq_len, num_views, embed_dim)

        # Apply multi-head attention across the views
        # Shape: [batch_size * seq_len, num_views, embed_dim]
        attn_output, _ = self.multihead_attn(x, x, x)

        # Reshape back to [batch_size, num_views, seq_len, embed_dim]
        attn_output = attn_output.view(
            batch_size, seq_len, num_views, embed_dim
            )
        # Shape: [batch_size, num_views, seq_len, embed_dim]
        attn_output = attn_output.permute(0, 2, 1, 3)

        return attn_output
