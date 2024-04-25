from torch import nn
import torch

from src.neural_decoder.augmentations import GaussianSmoothing


class ResnetDecoder(nn.Module):
    def __init__(
        self,
        n_classes: int,
        input_channels: int,
        neural_dim,
        gaussianSmoothWidth,
        dropout,
        nDays,
    ) -> None:
        super().__init__()

        # Neural decoder trainer expects model to have those
        self.kernelLen = 0
        self.strideLen = 1

        self.transform_to_3_channel = nn.Conv2d(
            input_channels, 3, kernel_size=3, stride=1, padding=1
        )

        self.resnet = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet152", pretrained=True
        )
        self.resnet.eval()

        resnet_output_size = 1000

        self.fc_decoder_out = nn.Linear(resnet_output_size, n_classes)

        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.dropout = nn.Dropout(dropout)

    def forward(self, neural_input, day_idx):
        neural_input = torch.permute(neural_input, (0, 2, 1))
        neural_input = self.gaussianSmoother(neural_input)
        neural_input = torch.permute(neural_input, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, day_idx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neural_input, dayWeights
        ) + torch.index_select(self.dayBias, 0, day_idx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        batch_size = transformedNeural.shape[0]
        seq_len = transformedNeural.shape[1]
        reshaped = transformedNeural.view(-1, 1, 16, 16)  # 4, 8, 8)
        # Resnet expects 3 channels
        out = self.transform_to_3_channel(reshaped)
        out = self.resnet(out)
        out = self.dropout(out)

        # Get phoneme probabilities
        out = self.fc_decoder_out(out)
        return out.view(batch_size, seq_len, -1)
