from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np

class ResidualBlock(nn.Module):
    """Configurable residual block
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : list of int
        Number of channels in the internal feature maps. Can either have two or three elements: if three construct
        a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
        `3 x 3` then `1 x 1` convolutions.
    stride : int
        Stride of the first `3 x 3` convolution
    dilation : int
        Dilation to apply to the `3 x 3` convolutions.
    groups : int
        Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
        bottleneck blocks.
    norm_act : callable
        Function to create normalization / activation Module.
    dropout: callable
        Function to create Dropout Module.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm_act=nn.BatchNorm2d,
                 dropout=None,
                 last = False): # 表示最后一层

        super(ResidualBlock, self).__init__()

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        if not is_bottleneck:
            bn2 = norm_act(channels[1])
            bn2.activation = "identity"
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 3, stride=stride, padding=dilation, bias=False,
                                    dilation=dilation)),
                ("bn1", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False,
                                    dilation=dilation)),
                ("bn2", bn2)
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]

        else:
            bn3 = norm_act(channels[2])
            bn3.activation = "identity"
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 1, stride=1, padding=0, bias=False)),
                ("bn1", norm_act(channels[0])),
                ("extra1", nn.LeakyReLU(0.01)), # zhangcb change

                ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=stride, padding=dilation, bias=False,
                                    groups=groups, dilation=dilation)),
                ("bn2", norm_act(channels[1])),

                ("dropout", nn.Dropout()),               

                ("extra2", nn.LeakyReLU(0.01)), # zhangcb change
                ("conv3", nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False)),
                ("bn3", bn3),

                ("conv2_new", nn.Conv2d(channels[0], channels[1], 3, stride=stride, padding=dilation, bias=False,
                                    groups=groups, dilation=dilation)),
                ("bn2_new", norm_act(channels[1])),

                ("dropout_new", nn.Dropout())
            ]
            # if dropout is not None:
                # layers = layers[0:5] + [("dropout", dropout())] + layers[4:]
            self.dropout = dropout
        self.convs = nn.Sequential(OrderedDict(layers))
        
        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)
            self.proj_bn = norm_act(channels[-1])
            self.proj_bn.activation = "identity"


        self._last = last

        # self.alpha = torch.nn.Parameter(torch.FloatTensor(2, channels[1]), requires_grad=True)
        # self.alpha.data[:,:] = 1.

    def forward(self, x):
        if hasattr(self, "proj_conv"):
            residual = self.proj_conv(x)
            residual = self.proj_bn(residual)
        else:
            residual = x
        # x = self.convs(x) + residual
        x = self.convs[0:3](x)

        x_branch1 = self.convs[3:5](x)
        if self.dropout is not None:
            x_branch1 = self.convs[5](x_branch1)
        x_branch2 = self.convs[9:11](x)
        if self.dropout is not None:
            x_branch2 = self.convs[11](x_branch2)
        
        
        r = torch.rand(1, x_branch1.shape[1], 1, 1, dtype=torch.float32)
        if self.training == False:
            r[:,:,:,:] = 1.0
        weight_out_branch = torch.zeros_like(r)
        weight_out_new_branch = torch.zeros_like(r)
        weight_out_branch[r < 0.33] = 2.
        weight_out_new_branch[r < 0.33] = 0.
        weight_out_branch[(r < 0.66)*(r>=0.33)] = 0.
        weight_out_new_branch[(r < 0.66)*(r>=0.33)] = 2.
        weight_out_branch[r>=0.66] = 1.
        weight_out_new_branch[r>=0.66] = 1.


        x = x_branch1 * weight_out_branch.to(x_branch1.device) * 0.5 + x_branch2 * weight_out_new_branch.to(x_branch1.device) * 0.5
        ######## random drop-path

        x = self.convs[6:9](x) + residual

        # all is leaky relu
        
        # if self.convs.bn1.activation == "leaky_relu":
        #     return functional.leaky_relu(x, negative_slope=self.convs.bn1.activation_param, inplace=True)
        # elif self.convs.bn1.activation == "elu":
        #     return functional.elu(x, alpha=self.convs.bn1.activation_param, inplace=True)
        # elif self.convs.bn1.activation == "identity":
        #     return x
        if self._last:
            return functional.leaky_relu_(x, negative_slope=0.01), x_branch1, x_branch2, x
        return functional.leaky_relu_(x, negative_slope=0.01)








# class ResidualBlock(nn.Module):
#     """Configurable residual block

#     Parameters
#     ----------
#     in_channels : int
#         Number of input channels.
#     channels : list of int
#         Number of channels in the internal feature maps. Can either have two or three elements: if three construct
#         a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
#         `3 x 3` then `1 x 1` convolutions.
#     stride : int
#         Stride of the first `3 x 3` convolution
#     dilation : int
#         Dilation to apply to the `3 x 3` convolutions.
#     groups : int
#         Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
#         bottleneck blocks.
#     norm_act : callable
#         Function to create normalization / activation Module.
#     dropout: callable
#         Function to create Dropout Module.
#     """

#     def __init__(self,
#                  in_channels,
#                  channels,
#                  stride=1,
#                  dilation=1,
#                  groups=1,
#                  norm_act=nn.BatchNorm2d,
#                  dropout=None):
#         super(ResidualBlock, self).__init__()

#         # Check parameters for inconsistencies
#         if len(channels) != 2 and len(channels) != 3:
#             raise ValueError("channels must contain either two or three values")
#         if len(channels) == 2 and groups != 1:
#             raise ValueError("groups > 1 are only valid if len(channels) == 3")

#         is_bottleneck = len(channels) == 3
#         need_proj_conv = stride != 1 or in_channels != channels[-1]

#         if not is_bottleneck:
#             bn2 = norm_act(channels[1])
#             bn2.activation = "identity"
#             layers = [
#                 ("conv1", nn.Conv2d(in_channels, channels[0], 3, stride=stride, padding=dilation, bias=False,
#                                     dilation=dilation)),
#                 ("bn1", norm_act(channels[0])),
#                 ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False,
#                                     dilation=dilation)),
#                 ("bn2", bn2)
#             ]
#             if dropout is not None:
#                 layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
#         else:
#             bn3 = norm_act(channels[2])
#             bn3.activation = "identity"
#             layers = [
#                 ("conv1", nn.Conv2d(in_channels, channels[0], 1, stride=1, padding=0, bias=False)),
#                 ("bn1", norm_act(channels[0])),
#                 ("extra1", nn.LeakyReLU(0.01)), # zhangcb change
#                 ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=stride, padding=dilation, bias=False,
#                                     groups=groups, dilation=dilation)),
#                 ("bn2", norm_act(channels[1])),
#                 ("extra2", nn.LeakyReLU(0.01)), # zhangcb change
#                 ("conv3", nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False)),
#                 ("bn3", bn3)
#             ]
#             if dropout is not None:
#                 layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
#         self.convs = nn.Sequential(OrderedDict(layers))
        
#         if need_proj_conv:
#             self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)
#             self.proj_bn = norm_act(channels[-1])
#             self.proj_bn.activation = "identity"

#     def forward(self, x):
#         if hasattr(self, "proj_conv"):
#             residual = self.proj_conv(x)
#             residual = self.proj_bn(residual)
#         else:
#             residual = x
#         x = self.convs(x) + residual       

#         # 这里全部都是leaky_relu
        
#         # if self.convs.bn1.activation == "leaky_relu":
#         #     return functional.leaky_relu(x, negative_slope=self.convs.bn1.activation_param, inplace=True)
#         # elif self.convs.bn1.activation == "elu":
#         #     return functional.elu(x, alpha=self.convs.bn1.activation_param, inplace=True)
#         # elif self.convs.bn1.activation == "identity":
#         #     return x
#         return functional.leaky_relu_(x, negative_slope=0.01)


class IdentityResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm_act=nn.BatchNorm2d,
                 dropout=None):
        """Configurable identity-mapping residual block
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps. Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        """
        super(IdentityResidualBlock, self).__init__()

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 3, stride=stride, padding=dilation, bias=False,
                                    dilation=dilation)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False,
                                    dilation=dilation))
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 1, stride=stride, padding=0, bias=False)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False,
                                    groups=groups, dilation=dilation)),
                ("bn3", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False))
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        if hasattr(self, "proj_conv"):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)

        out = self.convs(bn1)
        out.add_(shortcut)

        return out
