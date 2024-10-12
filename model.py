import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from torch import nn

from pet_seg_train.config import PetSegTrainConfig


class DoubleConvOriginal(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvOriginal, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConvDepthwiseSep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvDepthwiseSep, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(pl.LightningModule):
    def __init__(self, in_channels, out_channels, channels_list=[64, 128, 256, 512], depthwise_sep=False):
        super(UNet, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if depthwise_sep:
            DoubleConv = DoubleConvDepthwiseSep
        else :
            DoubleConv = DoubleConvOriginal
        

        # Encoder
        for channels in channels_list:
            self.encoder.append(DoubleConv(in_channels, channels))
            in_channels = channels

        # Decoder
        for channels in channels_list[::-1]:
            self.decoder.append(
                nn.ConvTranspose2d(channels * 2, channels, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(channels * 2, channels))

        self.bottleneck = DoubleConv(channels_list[-1], channels_list[-1] * 2)
        self.out = nn.Conv2d(channels_list[0], out_channels, kernel_size=1)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        skip_connections = []
        for i, enc_block in enumerate(self.encoder):
            x = enc_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip_connection = skip_connections[i // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat(
                (skip_connection, x), dim=1
            )  # Concatenate along the channel dimension
            x = self.decoder[i + 1](concat_skip)

        x = self.out(x)

        return x

    def training_step(self, batch, batch_idx):
        prefix = "train"
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, (y * 255 - 1).long().squeeze(1))
        self.log(f"{prefix}_loss", loss.item(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=PetSegTrainConfig.LEARNING_RATE)
