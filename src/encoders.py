import torch
import math
import torch.nn.functional as F


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


class PatchAttentionPooling(torch.nn.Module):

    """
    Attention-like pooling over patch embeddings.

    Input: [batch_size, patch_count, embedding_dim]
    Output: [batch_size, embedding_dim]
    """

    def __init__(self, dim: int):

        """
        Parameters:
            dim (int): Dimensionality of each patch embedding.
        """

        super().__init__()

        self.score = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.Tanh(),
            torch.nn.Linear(dim, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        """
        Pool patch embeddings into one image embedding.

        Parameters:
            x (torch.Tensor): Patch embeddings of shape [batch_size, patch_count, dim].

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - pooled: Tensor of shape [batch_size, dim].
                - attention_weights: Tensor of shape [batch_size, patch_count].
        """

        # compute one scalar score per patch
        scores = self.score(x).squeeze(-1)  # [B, P]

        # convert scores into normalized weights
        attention_weights = torch.softmax(scores, dim=1)  # [B, P]

        # weighted sum across patches
        pooled = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)  # [B, D]

        return pooled, attention_weights

class Encoder(torch.nn.Module):

    """
    Encoder for patched text images

    Expected input: [batch_size, patch_count, channels, height, width]

    Processing steps:
        1. Encode each patch independently with CNN + transformer.
        2. Obtain one embedding per patch.
        3. Aggregate patch embeddings into one image-level embedding.

    Output: [batch_size, dim]
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

        # projection from patch-level transformer feature to patch embedding
        self.patch_output_layer = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, dim)
        )

        # patch aggregation module.
        self.patch_pool = PatchAttentionPooling(dim=dim)

    def encode_single_patch_batch(self, x: torch.Tensor) -> torch.Tensor:

        """
        Encode a batch of individual patches.

        Parameters:
            x (torch.Tensor): Tensor of shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Patch embeddings of shape [batch_size, dim].
        """

        # step 1: local visual feature extraction
        x = self.conv(x)  # [B, 512, H', W']
        x = x[:, :, 0]  # [B, 512, W']

        # transformer expects [seq_len, batch_size, embedding_dim]
        x = x.permute(2, 0, 1)  # [W', B, 512]

        # add positional information along the sequence dimension
        x = self.pos_enc(x)

        # Step 2: sequence modeling across width
        x = self.transformer_encoder(x)  # [W', B, 512]
        x = x.permute(1, 2, 0)  # [B, 512, W'] - convert back to [batch_size, channels, seq_len] for pooling

        # Step 3: reduce sequence to one feature vector per patch.
        x = F.adaptive_avg_pool1d(x, 1)  # [B, 512, 1]
        x = x[:, :, 0]  # [B, 512] - remove the final singleton sequence dimension

        # Step 4: project to patch embedding space
        x = self.patch_output_layer(x)  # [B, dim]

        # normalize patch embeddings.
        x = F.normalize(x, dim=1)

        return x

    def forward(
        self,
        x: torch.Tensor,
        return_patch_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        """
        Encode patched images into one embedding per original image.

        Parameters:
             x (torch.Tensor): Tensor of shape [batch_size, patch_count, channels, height, width].
             return_patch_weights (bool, default=False): if True, also return the learned patch attention weights

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                - if return_patch_weights=False: Tensor of shape [batch_size, dim]
                - if return_patch_weights=True: (
                      image_embeddings: [batch_size, dim],
                      patch_weights:    [batch_size, patch_count]
                  )

        Raises:
            ValueError: if input shape is invalid
        """

        if x.ndim != 5:
            raise ValueError(
                f"Expected input of shape [batch_size, patch_count, channels, height, width], got {x.shape}"
            )

        batch_size, patch_count, channels, height, width = x.shape

        # flatten patches into a standard image batch.
        x = x.view(batch_size * patch_count, channels, height, width)

        # encode each patch independently.
        patch_embeddings = self.encode_single_patch_batch(x)  # [B * P, D]

        # restore patch structure
        patch_embeddings = patch_embeddings.view(batch_size, patch_count, -1)  # [B, P, D]

        # aggregate patch embeddings into one image embedding.
        image_embeddings, patch_weights = self.patch_pool(patch_embeddings)  # [B, D], [B, P]

        # normalize final image embeddings again after pooling.
        image_embeddings = F.normalize(image_embeddings, dim=1)

        if return_patch_weights:
            return image_embeddings, patch_weights

        return image_embeddings
