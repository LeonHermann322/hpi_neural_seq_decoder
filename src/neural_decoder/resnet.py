from torch import nn
import torch


class ResnetDecoder(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()

        # Neural decoder trainer expects model to have those
        self.kernelLen = 0
        self.strideLen = 1

        self.transform_to_3_channel = nn.Conv2d(
            4, 3, kernel_size=3, stride=1, padding=1
        )

        self.resnet = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet152", pretrained=True
        )

        self.resnet.train()

        resnet_output_size = 1000

        self.fc_decoder_out = nn.Linear(resnet_output_size, n_classes)

    def forward(self, neural_input, day_idx):
        batch_size = neural_input.shape[0]
        seq_len = neural_input.shape[1]
        reshaped = neural_input.view(-1, 4, 8, 8)
        # Resnet expects 3 channels
        out = self.transform_to_3_channel(reshaped)
        out = self.resnet(out)

        # Get phoneme probabilities
        out = self.fc_decoder_out(out)
        return out.view(batch_size, seq_len, -1)
