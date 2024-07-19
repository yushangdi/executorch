from dataclasses import dataclass

from typing import List, Optional

import torch

from executorch.examples.models.model_base import EagerModelBase
from torch.export import Dim
from torchvision.transforms import v2
from torchvision.transforms._functional_tensor import resize


@torch.library.custom_op("preprocess::pad", mutates_args=())
def pad(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y.narrow(1, 0, x.shape[1]).narrow(2, 0, x.shape[2]).copy_(x)
    return y.clone()


@torch.library.register_fake("preprocess::pad")
def pad(x, y):
    return torch.empty_like(y)


@torch.library.custom_op("preprocess::reshape", mutates_args=())
def reshape(c: int, tile_size: int, output: torch.Tensor) -> torch.Tensor:
    h = output.shape[1]
    w = output.shape[2]
    tiles_height = h // tile_size
    tiles_width = w // tile_size
    reshaped = output.view(c, tiles_height, tile_size, tiles_width, tile_size)
    transposed = reshaped.permute(1, 3, 0, 2, 4)
    tiles = transposed.contiguous().view(
        tiles_height * tiles_width, c, tile_size, tile_size
    )
    return tiles


@torch.library.register_fake("preprocess::reshape")
def reshape(c: int, tile_size: int, output: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(output)


@dataclass
class PreprocessConfig:
    tile_size: int
    channels: int
    image_mean: List[float]
    image_std: List[float]
    resample: str
    normalize: bool


class Preprocess(torch.nn.Module):
    def __init__(self, config: Optional[PreprocessConfig] = None):
        super().__init__()
        if config is None:
            self.config = PreprocessConfig(
                tile_size=224,
                channels=3,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711],
                resample="bilinear",
                normalize=True,
            )
        else:
            self.config = config

    def forward(
        self, image: torch.Tensor, target_size: torch.Tensor, canvas_size: torch.Tensor
    ):
        # Resize
        ts0, ts1 = target_size.tolist()
        torch._check(ts0 >= 2)
        torch._check(ts0 <= 4000)
        torch._check(ts1 >= 2)
        torch._check(ts1 <= 4000)

        image = resize(
            image,
            size=[ts0, ts1],
            interpolation=self.config.resample,
        )

        # Pad
        cs0, cs1 = canvas_size.tolist()
        torch._check(cs0 >= 2)
        torch._check(cs0 <= 4000)
        torch._check(cs1 >= 2)
        torch._check(cs1 <= 4000)
        sizes = [3, cs0, cs1]

        output = torch.empty(
            sizes, dtype=image.dtype, device=image.device, requires_grad=False
        )
        output = torch.fill(output, 0)
        output = torch.ops.preprocess.pad(image, output)

        # Normalize
        if self.config.normalize:
            output = v2.functional.normalize(
                output, self.config.image_mean, self.config.image_std
            )

        # Split
        tiles = torch.ops.preprocess.reshape(
            self.config.channels, self.config.tile_size, output
        )
        return tiles


class PreprocessModel(EagerModelBase):
    def __init__(self):
        super().__init__()

    def get_eager_model(self):
        model = Preprocess()
        return model

    def get_example_inputs(self):
        image = torch.ones(3, 800, 600)
        target_size = torch.tensor([448, 336])
        canvas_size = torch.tensor([448, 448])
        return (image, target_size, canvas_size)

    def get_dynamic_shapes(self):
        img_h = Dim("img_h", min=1, max=4000)
        img_w = Dim("img_w", min=1, max=4000)

        dynamic_shapes = {
            "image": {1: img_h, 2: img_w},
            "target_size": None,
            "canvas_size": None,
        }
        return dynamic_shapes
