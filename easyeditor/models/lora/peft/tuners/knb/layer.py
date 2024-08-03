# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .config import KnbConfig


class KnbLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("knb_W", "knb_embedding_W")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("knb_alpha", "scaling", "knb_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.length = {} # cjc@0802 add length dict key: adapter_name, value: length/knb_idx_list
        self.knb_alpha = {}
        self.scaling = {}
        self.knb_dropout = nn.ModuleDict({})
        self.knb_W = nn.ModuleDict({})
        # For Embedding layer
        self.knb_embedding_W = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self._caches: dict[str, Any] = {}
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name, length, knb_alpha, knb_dropout, init_knb_weights, use_rsknb
    ):
        # This code works for linear layers, override for other layer types
        # TODO:lzc@0802 length=0 torch.empty(0,0)
        # length<=0,都不会运行到这里
        if length <= 0:
            # length <= 0 means no KNB, return
            return 
            # raise ValueError(f"`length` should be a positive integer value but the value passed is {length}")

        self.length[adapter_name] = length # TODO: cjc@0802 add length dict key: adapter_name, value: length/knb_idx_list
        self.knb_alpha[adapter_name] = knb_alpha
        if knb_dropout > 0.0:
            knb_dropout_layer = nn.Dropout(p=knb_dropout)
        else:
            knb_dropout_layer = nn.Identity()

        self.knb_dropout.update(nn.ModuleDict({adapter_name: knb_dropout_layer}))
        # Actual trainable parameters
        # TODO:lzc@0802
        # # knb_W: [in_features, length] length -> out_features knb
        # self.knb_W[adapter_name] = nn.Linear(self.in_features, length, bias=False)
        # 知识神经元论文中的梯度归因方法,定位到的kn_idx是下一层的in_features
        # knb_W: [lenght, out_features] length -> in_features knb
        self.knb_W[adapter_name] = nn.Linear(length, self.out_features, bias=False)
        if use_rsknb:
            self.scaling[adapter_name] = knb_alpha / math.sqrt(length)
        else:
            self.scaling[adapter_name] = knb_alpha / length


        if init_knb_weights:
            self.reset_knb_parameters(adapter_name, init_knb_weights)

        self.set_adapter(self.active_adapters)

    def reset_knb_parameters(self, adapter_name, init_knb_weights):
        if init_knb_weights is False:
            return

        if adapter_name in self.knb_W.keys():
            if init_knb_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/knblib/layers.py#L124
                nn.init.kaiming_uniform_(self.knb_W[adapter_name].weight, a=math.sqrt(5))
            elif init_knb_weights.lower() == "gaussian":
                nn.init.normal_(self.knb_W[adapter_name].weight, std=1 / self.length[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_knb_weights=}")
            
        if adapter_name in self.knb_embedding_W.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.knb_embedding_W[adapter_name])

    def _get_weight_norm(self, weight, knb_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = transpose(weight, self.fan_in_fan_out)
        weight = weight + scaling * knb_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.knb_alpha[adapter] / self.length[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.knb_W.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.knb_W.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.knb_alpha[active_adapter] / self.length[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        unique_adapters = set(self.active_adapters)

# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, KnbLayer):
    # Knb implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        knb_alpha: int = 1,
        knb_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_knb_weights: Union[bool, str] = True,
        use_rsknb: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        KnbLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        # add knowledge neurons idx list cjc@0802
        self.kn_idx_list = None
        if kwargs.get("target_name") and kwargs.get("knb_dict"):
            target_name=kwargs.get("target_name") # 'transformer.h.0.mlp.c_proj' 'model.layers.0.mlp.up_proj
            knb_dict=kwargs.get("knb_dict")
            # "knb dict"
            # "knb_dict:key1=weight_name, value1=kn_dict"
            # "kn_dict:key2=layer_idx, value2=kn_idx_list"
            weight_name = ".".join(target_name.split(".")[-2:])
            layer_idx = target_name.split(".")[2]
            if weight_name in knb_dict and layer_idx in knb_dict[weight_name]:
                kn_idx_list = knb_dict[weight_name][layer_idx]
                self.kn_idx_list = kn_idx_list if len(kn_idx_list) > 0 else None
        else:
            print("No target_name or knb_dict found in kwargs")
        # TODO: lzc@0802 length=0 may cause bug
        length = len(self.kn_idx_list) if self.kn_idx_list is not None else 0
        self.update_layer(
            adapter_name,
            length,
            knb_alpha=knb_alpha,
            knb_dropout=knb_dropout,
            init_knb_weights=init_knb_weights,
            use_rsknb=use_rsknb,
        )
    def get_delta_weight(self, active_adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.knb_W[active_adapter].weight.device
        dtype = self.knb_W[active_adapter].weight.dtype
        # nn.Linear weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        delta_W = torch.zeros(self.out_features, self.in_features, device=device, dtype=dtype)
        dim1, dim2 = self.knb_W[active_adapter].weight.shape
        if self.kn_idx_list is not None:
            if dim1 < self.out_features and dim2 == self.in_features:
                delta_W[self.kn_idx_list, :] = self.knb_W[active_adapter].weight
            elif dim2 < self.in_features and dim1 == self.out_features:
                delta_W[:, self.kn_idx_list] = self.knb_W[active_adapter].weight
            return delta_W
        else:
            return delta_W # 注意这里返回的是全0张量


    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.knb_W.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights = orig_weights + delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data = base_layer.weight.data + delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.knb_W.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data -= delta_weight


    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.knb_W.keys():
                    continue
                knb_W = self.knb_W[active_adapter]
                dropout = self.knb_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(knb_W.weight.dtype)
                # 维度问题
                if self.kn_idx_list is not None:
                    delta_w = self.get_delta_weight(active_adapter) # [out_features, in_features]
                    delta_w = delta_w.to(dtype=x.dtype)
                    result += scaling * (dropout(x) @ delta_w.T) # [batch_size, seq_len, in_features]@[in_features, out_features] = [batch_size, seq_len, out_features]

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "knb." + rep

def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    knb_config: KnbConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = knb_config.fan_in_fan_out = False
        # add knb_dict to kwargs 0422@cjc
        if knb_config.knb_dict:
            kwargs["knb_dict"] = knb_config.knb_dict
            # print("knb_dict is added to kwargs")
        new_module = Linear(target, adapter_name, **kwargs)

    return new_module
