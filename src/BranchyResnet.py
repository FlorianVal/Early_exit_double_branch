import re

import torch
from torch import Tensor, nn
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck
from matplotlib import pyplot as plt
import torchvision


class AuxiliaryHead(nn.Module):
    """ Class to define Auxiliary head in a branchy network.
    """

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, num_classes)
        self.fc_reg = nn.Linear(in_features, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward method of the auxiliary head.
        """
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.fc_reg(x)
        x = self.fc(x)
        # x = self.softmax(x)
        return x, y


class BranchyResNet(ResNet):
    """Implementation of a Branchy Resnet : a Resnet with auxiliary output heads at each layer"""

    def __init__(self, num_classes: int = 1000, num_aux_heads: int = 4, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs)
        # add auxiliary heads to each layer
        self.softmax = nn.Softmax(dim=1)
        self.main_block = []
        self.aux_head_output = None
        self.cached_features = None
        self.previous_batch = None
        total_layers = 0
        for group in re.findall(r'layer\d', str(self)):
            total_layers += len(getattr(self, group))
        aux_head_interval = total_layers // (num_aux_heads + 1)
        aux_head_counter = 0
        for group in re.findall(r'layer\d', str(self)):
            for layer in getattr(self, group):
                if isinstance(layer, (BasicBlock, Bottleneck)):
                    self.main_block.append(layer)
                    if aux_head_counter < num_aux_heads and len(self.main_block) % aux_head_interval == 0:
                        self.main_block.append(AuxiliaryHead(
                            getattr(self, group)[-1].conv1.in_channels, num_classes))
                        aux_head_counter += 1
                else:
                    self.main_block.append(layer)

        self.main_block = nn.Sequential(*self.main_block)
        del self.layer1
        del self.layer2
        del self.layer3
        del self.layer4


    def _forward_impl(self, x: Tensor) -> Tensor:
        """forward pass

        Returns:
            Tensor: tensor of shape [n_head, batch_size, num_classes]
        """
        # overwrite forward method of ResNet
        # create another forward fot inference with reject

        def forward_output_layer(x):
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            # x = self.softmax(x)
            return x

        def forward_input_layer(x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            return x

        def forward_with_aux_head(layers, x, aux_outputs, reg_outputs, aux_head_output=None, non_filtered_index=None):
            if aux_head_output is not None:
                # will output one head
                aux_head_count = 0
                for layer_index, layer in enumerate(layers):
                    if non_filtered_index is not None and self.cached_features is not None and self.cached_features["rank"] < layer_index:
                        print("skipping layer")
                        continue
                    if non_filtered_index is not None and self.cached_features is not None and self.cached_features["rank"] == layer_index:
                        # resume inference
                        x = self.cached_features["data"]
                        x = x[torch.BoolTensor(non_filtered_index)]
                    if isinstance(layer, AuxiliaryHead):
                        aux_head_count += 1
                        if aux_head_count == aux_head_output:
                            self.cached_features = {
                                "data": x, "rank": layer_index}
                            aux_outputs, reg_outputs = layer(x)
                            return aux_outputs.unsqueeze(0), reg_outputs.unsqueeze(0)
                    else:
                        x = layer(x)
                x = forward_output_layer(x)
                return x
            else:
                # will output every head predictions
                for layer in layers:
                    if isinstance(layer, AuxiliaryHead):
                        auxiliary_head_output, reg_value = layer(x)
                        aux_outputs = torch.cat(
                            (aux_outputs, auxiliary_head_output.unsqueeze(0)), dim=0)
                        reg_outputs = torch.cat((reg_outputs, reg_value.unsqueeze(0)), dim=0)
                    else:
                        x = layer(x)
                x = forward_output_layer(x)
                return torch.cat((x.unsqueeze(0), aux_outputs), dim=0), reg_outputs

        aux_outputs = torch.empty(0).to(x.device)
        reg_outputs = torch.empty(0).to(x.device)

        x = forward_input_layer(x)
        return forward_with_aux_head(self.main_block, x, aux_outputs, reg_outputs, self.aux_head_output)

    def set_aux_head_output(self, aux_head_output: int):
        """ set the auxiliary head for inference

        Args:
            aux_head_output (int): head number to use for inference
        """
        if aux_head_output is not None:
            if aux_head_output == -1:
                aux_head_output = len(self.main_block) // 2 + 1
            assert aux_head_output <= len(
                self.main_block) // 2 + 1, f"aux_head_output must be less than {len(self.main_block) // 2 + 1}, got {aux_head_output}"
            assert aux_head_output > 0, f"head indexing start at 1, got {aux_head_output}"
        self.aux_head_output = aux_head_output


class BranchyResNet18(BranchyResNet):
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs,
                         block=BasicBlock, layers=[2, 2, 2, 2])


class BranchyResNet34(BranchyResNet):
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs,
                         block=BasicBlock, layers=[3, 4, 6, 3])


class BranchyResNet50(BranchyResNet):
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs,
                         block=Bottleneck, layers=[3, 4, 6, 3])


class BranchyResNet101(BranchyResNet):
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs,
                         block=Bottleneck, layers=[3, 4, 23, 3])


class BranchyResNet152(BranchyResNet):
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs,
                         block=Bottleneck, layers=[3, 8, 36, 3])


def show_image_grid(images, nrow):
    grid = image_grid(images.cpu(), nrow=12)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()


def image_grid(images, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        images (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Returns:
        Tensor: 3D grid of images with shape (C x H x W).
    """
    if not (torch.is_tensor(images) or
            (isinstance(images, list) and all(torch.is_tensor(t) for t in images))):
        raise TypeError('images should be Tensor or a list of Tensor')

    if isinstance(images, list):
        images = torch.stack(images, dim=0)

    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    if images.shape[0] == 1:
        return images[0]

    return torchvision.utils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize,
                                       range=range, scale_each=scale_each, pad_value=pad_value)