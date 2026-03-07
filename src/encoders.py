import torch
import math


def create_vgg_block(input_channels, output_channels, subsampling=(2, 2)):
    return torch.nn.Sequential(
        torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        torch.nn.LeakyReLU(),
        # torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        # torch.nn.LeakyReLU(),
        torch.nn.MaxPool2d(kernel_size=subsampling, stride=subsampling),
        torch.nn.InstanceNorm2d(num_features=output_channels)
    )


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 60):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]


class Encoder(torch.nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.conv = torch.nn.Sequential(
            create_vgg_block(3, 32, subsampling=(2, 2)),
            create_vgg_block(32, 64, subsampling=(2, 2)),
            create_vgg_block(64, 128, subsampling=(2, 2)),
            torch.nn.Conv2d(128, 512, kernel_size=(5, 1)),
            torch.nn.LeakyReLU(),
        )
        self.pos_enc = PositionalEncoding(d_model=512, max_len=100)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=512, dim_feedforward=512, nhead=4)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.output_layer = torch.nn.Sequential(torch.nn.LeakyReLU(), torch.nn.Linear(512, dim))

    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, 0].permute(2, 0, 1)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)
        x = torch.nn.functional.adaptive_avg_pool1d(x, 1)
        x = x[:, :, 0]
        x = self.output_layer(x)
        x = torch.nn.functional.normalize(x)
        return x
