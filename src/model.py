import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchCNN(nn.Module):

    """
    Shared CNN backbone that encodes a single image patch into a feature vector.

    Four convolutional blocks extract local visual features (strokes, textures, pen pressure).
    Blocks 1–3 each halve the spatial dimensions via MaxPool2d (8x total downsampling).
    Block 4 refines features at the reduced spatial resolution without further downsampling, and outputs hidden_dim channels directly.
    Global average pooling then collapses the remaining spatial dimensions into a single vector regardless of the input patch size.

    The weights are shared across all patches.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 256):

        """
        Initialize CNN backbone for patches.

        Parameters:
            in_channels (int, default=3): Number of input channels.
            hidden_dim (int, default=256): Output feature dimensionality. Block 4 outputs hidden_dim channels directly.
        """

        super().__init__()

        self.blocks = nn.Sequential(
            # Block 1: coarse edge and texture detection (spatial: H -> H / 2)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: mid-level stroke and shape features (spatial: H / 2 -> H / 4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: higher-level character part features (spatial: H / 4 -> H / 8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(0.1),  # spatial dropout for CNN regularization

            # Block 4: abstract feature refinement - no spatial reduction.
            nn.Conv2d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        # collapse remaining spatial dimensions to a single value per channel
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Encode a batch of patches into feature vectors.

        Parameters:
            x (torch.Tensor): Batch of image patches with shape [B, C, H, W], where B is the number of patches.

        Returns:
            torch.Tensor: Patch embeddings with shape [B, hidden_dim].
        """

        x = self.blocks(x)  # [B, hidden_dim, H / 8, W / 8]
        x = self.gap(x)     # [B, hidden_dim, 1, 1]
        x = x.flatten(1)    # [B, hidden_dim]

        return x


class MultiHeadAttentionPooling(nn.Module):

    """
    Multi-head attention-based aggregation of N patch embeddings into a single image embedding.

    Each head independently scores all N patches and computes a weighted sum.
    Multiple heads can specialise on different handwriting aspects - one head may focus on slant, another on pen pressure, another on letter spacing, and so on.
    Head outputs are projected to head_dim = hidden_dim // num_heads each, then concatenated and projected back to hidden_dim, keeping the output shape identical to single-head pooling.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:

        """
        Initialize MultiHeadAttentionPooling.

        Parameters:
            hidden_dim (int): Dimensionality of the input patch embeddings. Must be divisible by num_heads.
            num_heads (int, default=4): Number of independent attention heads.
        """

        super().__init__()

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # one MLP scorer per head - maps each patch embedding to a scalar score
        self.scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_heads)
        ])

        # each head projects its weighted-sum output down to head_dim before concatenation
        self.head_projs = nn.ModuleList([
            nn.Linear(hidden_dim, self.head_dim)
            for _ in range(num_heads)
        ])

        # restore hidden_dim from the concatenated head outputs
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        """
        Pool N patch embeddings into a single image embedding using multi-head attention.

        Parameters:
            x (torch.Tensor): Patch embeddings with shape [B, N, hidden_dim].
            padding_mask (torch.Tensor, optional, default=None): boolean mask with shape [B, N].
                True = padded position (ignored), False = real patch.
                When provided, padded positions receive -inf scores before softmax so they contribute zero attention weight.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - pooled: Image embedding with shape [B, hidden_dim].
                - weights: Per-head attention weights with shape [B, num_heads, N].
        """

        head_outputs = []
        all_weights = []

        for scorer, head_proj in zip(self.scorers, self.head_projs):
            scores = scorer(x).squeeze(-1)  # [B, N]

            # mask padded positions so they get zero weight after softmax
            if padding_mask is not None:
                scores = scores.masked_fill(padding_mask, float('-inf'))

            weights = F.softmax(scores, dim=-1)                     # [B, N]
            pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [B, hidden_dim]
            head_outputs.append(head_proj(pooled))                  # [B, head_dim]
            all_weights.append(weights)

        concatenated = torch.cat(head_outputs, dim=-1)  # [B, hidden_dim]
        pooled = self.out_proj(concatenated)            # [B, hidden_dim]
        weights = torch.stack(all_weights, dim=1)       # [B, num_heads, N]

        return pooled, weights


class WriterIdentificationEncoder(nn.Module):

    """
    Full encoder for writer identification from patched handwriting images.

    Each input image is represented as a set of N patches produced by one of the available patching methods (grid, random, SIFT).

    The encoder processes these patches in three stages:

        1. PatchCNN - a shared CNN encodes every patch independently into a hidden_dim-dimensional
            feature vector, capturing local visual properties like stroke width, pen pressure, and character shapes.

        2. TransformerEncoder - 2 layers of multi-head self-attention allow every patch to attend
            to every other patch, building up cross-patch context.

        3. MultiHeadAttentionPooling - multiple learned MLP scorers independently decide which patches are most significant
            and produce a single image embedding as a weighted sum of the patch embeddings.
            Different heads can specialize on different handwriting aspects.

    The final embedding is projected to embed_dim dimensions and L2-normalized.

    Note on dimensionality:
        - for best metric learning behavior embed_dim should be <= hidden_dim (projecting down into a compact metric space).
            If embed_dim > hidden_dim, the projection is underconstrained - consider raising hidden_dim instead.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        nhead: int = 8,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_positional_encoding: bool = False,
        max_patches: int = 100,
        num_pool_heads: int = 4,
    ) -> None:

        """
        Initialize the WriterIdentification encoder.

        Parameters:
            in_channels (int, default=3): Number of input image channels.
            hidden_dim (int, default=256): Internal dimension shared by the CNN, the Transformer, and the attention pooling.
            embed_dim (int, default=128): Final output embedding dimension written into the metric space.
                                          Should be <= hidden_dim for a properly constrained projection. The user is expected to experiment with this value.
            nhead (int, default=8): Number of attention heads in each Transformer layer. Must divide hidden_dim evenly.
            num_transformer_layers (int, default=2): Number of stacked Transformer encoder layers.
            dim_feedforward (int, default=1024): Hidden dimension of the feed-forward network inside each Transformer layer.
            dropout (float, default=0.1): Dropout probability applied inside the Transformer.
            use_positional_encoding (bool, default=False): Whether to add sinusoidal positional encodings to the patch sequence before the Transformer.
                                                           Enable for grid-patching experiments where patches have a canonical spatial order.
                                                           Leave False for random and SIFT patches (no canonical order).
            max_patches (int, default=100): Maximum expected patch count. Only used to pre-compute the positional encoding buffer when use_positional_encoding=True.
            num_pool_heads (int, default=4): Number of independent attention heads in MultiHeadAttentionPooling. Must divide hidden_dim evenly.
        """

        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_positional_encoding = use_positional_encoding

        # Stage 1 - Patch-level CNN (input -> [B x N, in_channels, H, W]; output -> [B x N, hidden_dim])
        self.patch_cnn = PatchCNN(in_channels=in_channels, hidden_dim=hidden_dim)

        # optional sinusoidal positional encoding (for grid patching only)
        if use_positional_encoding:
            pe = self._build_sinusoidal_pe(max_patches, hidden_dim)  # [max_patches, hidden_dim]
            self.register_buffer("pe", pe)

        # Stage 2 - Cross-patch Transformer (input -> [B, N, hidden_dim]; output -> [B, N, hidden_dim])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
            enable_nested_tensor=False,
        )

        # Stage 3 - Multi-head attention pooling (input -> [B, N, hidden_dim]; output -> [B, hidden_dim] + weights [B, num_heads, N])
        self.pool = MultiHeadAttentionPooling(hidden_dim, num_heads=num_pool_heads)

        # Normalize pooled representation before projection for training stability
        self.pre_proj_norm = nn.LayerNorm(hidden_dim)

        # Output projection (input -> [B, hidden_dim]; output -> [B, embed_dim])
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_patch_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        """
        Encode a batch of patched handwriting images into normalized embeddings.

        Parameters:
            x (torch.Tensor): Patched images with shape [B, N, C, H, W], where B is the batch size, N is the number of patches per image.
            padding_mask (torch.Tensor | None): boolean mask with shape [B, N].
                True = padded position (ignored), False = real patch.
                Used by the grid patcher path where different images produce different numbers of patches and the batch is padded to the maximum count.
                When None, all patches are treated as real (random/sift patchers).
            return_patch_weights (bool, default=False): When True, also return the per-head per-patch attention weights from the pooling stage.
                Useful for visualizing which patches each head found most informative.

        Returns:
            torch.Tensor: L2-normalized image embeddings with shape [B, embed_dim].
            tuple[torch.Tensor, torch.Tensor]:
                (embeddings [B, embed_dim], patch_weights [B, num_pool_heads, N]) when return_patch_weights=True.
        """

        B, N, C, H, W = x.shape

        # Stage 1 - Encode every patch with the shared CNN
        x = x.reshape(B * N, C, H, W)  # [B x N, C, H, W]
        x = self.patch_cnn(x)          # [B x N, hidden_dim]
        x = x.reshape(B, N, -1)        # [B, N, hidden_dim]

        # optional positional encoding (grid patching only)
        if self.use_positional_encoding:
            x = x + self.pe[:N].unsqueeze(0)  # [1, N, hidden_dim] broadcast -> [B, N, hidden_dim]

        # Stage 2 - Cross-patch Transformer: every patch attends to every other patch
        # src_key_padding_mask tells the transformer which positions are padding (True = ignore)
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # [B, N, hidden_dim]

        # Stage 3 - Multi-head attention pooling: N embeddings -> 1 image embedding
        image_emb, patch_weights = self.pool(x, padding_mask=padding_mask)  # image_emb: [B, hidden_dim]; patch_weights: [B, num_heads, N]

        # Normalize + project + L2 normalize
        image_emb = self.pre_proj_norm(image_emb)       # [B, hidden_dim]
        image_emb = self.out_proj(image_emb)            # [B, embed_dim]
        image_emb = F.normalize(image_emb, p=2, dim=1)  # [B, embed_dim], ||emb|| = 1

        if return_patch_weights:
            return image_emb, patch_weights  # ([B, embed_dim], [B, num_heads, N])

        return image_emb  # [B, embed_dim]

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:

        """
        Build a fixed sinusoidal positional encoding matrix.

        Uses the standard Transformer positional encoding formula from "Attention Is All You Need"

        Parameters:
            max_len (int): Number of positions (maximum patch count) to pre-compute encodings for.
            d_model (int): Embedding dimensionality. Must be even.

        Returns:
            torch.Tensor: Positional encoding matrix with shape [max_len, d_model].
        """

        assert d_model % 2 == 0, f"d_model must be even for sinusoidal PE, got {d_model}"

        position = torch.arange(max_len).unsqueeze(1).float()  # [max_len, 1]

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # [d_model / 2]

        pe = torch.zeros(max_len, d_model)            # [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices - sine
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices - cosine

        return pe  # [max_len, d_model]
