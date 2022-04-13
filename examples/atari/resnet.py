# from https://github.com/facebookresearch/moolib/blob/e8b2de7ac5df3a9b3ee2548a33f61100a95152ef/examples/atari/models.py#L15

import torch
import torch.nn as nn


class ResNetEncoder(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        assert len(observation_space.shape) == 3
        input_channels = observation_space.shape[0]

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

    def forward(self, x):
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = x.view(T * B, -1)
        return x
