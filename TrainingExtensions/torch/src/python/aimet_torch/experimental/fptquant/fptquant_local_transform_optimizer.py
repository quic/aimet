# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-docstring

from aimet_torch.common.progress import progress_bar
import itertools
import torch

from aimet_torch.experimental.transforms.transformed_layers import TransformationMixin


class LocalTransformOptimizer:
    p: float = 4.0
    num_iterations: int = 200
    lr: float = 1e-2

    @staticmethod
    def compute_loss(weight: torch.Tensor) -> torch.Tensor:
        return torch.mean((weight.abs() ** LocalTransformOptimizer.p)) ** (
            1 / LocalTransformOptimizer.p
        )

    def __init__(self, transformed_layers: list[TransformationMixin]):
        trainable_parameters: dict[torch.nn.Module, set[torch.nn.Parameters]] = {}

        for layer in transformed_layers:
            for transform in itertools.chain(
                layer.right_hand_transforms, layer.left_hand_transforms
            ):
                trainable_parameters[layer] = set(
                    param
                    for param in transform.parameters()
                    if transform.mergeable and param.requires_grad
                )

        self.layers = [
            layer
            for layer, trainable_params in trainable_parameters.items()
            if trainable_params
        ]
        self.parameters = set.union(*trainable_parameters.values())
        self.optimizer = torch.optim.AdamW(self.parameters, lr=self.lr)

    # pylint: disable=protected-access
    def optimize(self):
        for _ in progress_bar(
            range(self.num_iterations), desc="Locally optimizing transforms"
        ):
            self.optimizer.zero_grad()
            with torch.nn.utils.parametrize.cached():
                loss = torch.stack(
                    tuple(
                        self.compute_loss(layer._compute_merged_params()[0])
                        for layer in self.layers
                    )
                ).sum(dim=0)
            loss.backward()
            self.optimizer.step()
