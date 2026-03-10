import torch
import math


def create_vgg_block(
    input_channels: int,
    output_channels: int,
    subsampling: tuple[int, int] = (2, 2)
) -> torch.nn.Sequential:

    """
    Create one convolutional block used in the encoder

    Parameters:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        subsampling (tuple[int, int]): Pooling kernel size and stride.

    Returns:
        torch.nn.Sequential: Convolutional block.
    """

    return torch.nn.Sequential(
        torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding=1
        ),
        torch.nn.LeakyReLU(),
        torch.nn.MaxPool2d(
            kernel_size=subsampling,
            stride=subsampling
        ),
        torch.nn.InstanceNorm2d(num_features=output_channels)
    )


class PositionalEncoding(torch.nn.Module):

    """
    Add sinusoidal positional encoding to sequence features.
    """

    def __init__(self, d_model: int, max_len: int = 60):

        """
        Initialize positional encoding.

        Parameters:
            d_model (int): Embedding dimension.
            max_len (int): Maximum supported sequence length.
        """

        super().__init__()

        # position indices: [0, 1, 2, ..., max_len - 1]
        position = torch.arange(max_len).unsqueeze(1)

        # frequency scaling term used in the sinusoidal encoding formula
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # positional encoding tensor of shape [max_len, 1, d_model]
        pe = torch.zeros(max_len, 1, d_model)

        # even dimensions use sine, odd dimensions use cosine
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # register as buffer so it moves with the model device,
        # but is not treated as a trainable parameter.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Add positional encoding to input sequence.

        Parameters:
            x (torch.Tensor): Input tensor of shape [seq_len, batch_size, embedding_dim].

        Returns:
            torch.Tensor: Input tensor with positional encoding added.
        """

        return x + self.pe[:x.size(0)]


class Encoder(torch.nn.Module):

    """
    Image encoder that combines CNN feature extraction, positional encoding, transformer sequence modeling, and final embedding projection.
    """

    def __init__(self, dim: int = 256):

        """
        Initialize encoder.

        Parameters:
            dim (int): Output embedding dimension.
        """

        super().__init__()

        # CNN backbone extracts local visual features from the input image.
        self.conv = torch.nn.Sequential(
            create_vgg_block(3, 32, subsampling=(2, 2)),
            create_vgg_block(32, 64, subsampling=(2, 2)),
            create_vgg_block(64, 128, subsampling=(2, 2)),
            torch.nn.Conv2d(128, 512, kernel_size=(5, 1)),
            torch.nn.LeakyReLU(),
        )

        # positional encoding is added before transformer processing
        self.pos_enc = PositionalEncoding(d_model=512, max_len=100)

        # transformer models relationships along the remaining sequence dimension
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=512,
            dim_feedforward=512,
            nhead=4
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        # final projection maps transformer features to the target embedding size
        self.output_layer = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Encode input image batch into normalized embedding vectors.

        Parameters:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Normalized embedding tensor of shape [batch_size, dim].
        """

        # Step 1: extract convolutional feature maps
        x = self.conv(x)
        x = x[:, :, 0]

        # transformer expects sequence-first format: [seq_len, batch_size, embedding_dim]
        # width becomes sequence length, channels become embedding dimension
        x = x.permute(2, 0, 1)

        # add positional information so transformer knows token order in the sequence
        x = self.pos_enc(x)

        # Step 2: model long-range dependencies across the sequence
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)  # convert back to [batch_size, channels, seq_len] for pooling.

        # Step 3: pool across the sequence dimension to get one vector per sample
        x = torch.nn.functional.adaptive_avg_pool1d(x, 1)
        x = x[:, :, 0]  # remove the final singleton sequence dimension.

        # Step 4: project pooled features to the final embedding space
        x = self.output_layer(x)

        x = torch.nn.functional.normalize(x)  # normalize embeddings to unit length (useful for similarity comparison with dot product / cosine similarity)

        return x
