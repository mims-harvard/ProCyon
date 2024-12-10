import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Union, List, Type

from peft.tuners.lora import LoraLayer
from peft.utils import (
    COMMON_LAYERS_PATTERN,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)
from peft import PeftModelForCausalLM, LoraModel, LoraConfig, PeftConfig
from peft.utils import _prepare_prompt_learning_config
from peft.tuners.lora import *
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaModel
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import logging

# from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

logger = logging.getLogger(__name__)

def create_experts(in_features: int, out_features: int, moe_num_experts = 4):
    return nn.ModuleList(
        [nn.Linear(in_features, out_features, bias=False) for _ in range(moe_num_experts)]
    )

@dataclass
class BaseMoEModelOutputWithPast(BaseModelOutputWithPast):
    router_logits: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MoECausalLMOutputWithPast(CausalLMOutputWithPast):
    router_logits: Optional[Tuple[torch.FloatTensor]] = None

def router_z_loss_func(router_logits: torch.Tensor) -> float:
    """From Huggingface SwitchTransformers"""
    num_groups, tokens_per_group, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z ** 2
    return torch.sum(z_loss) / (num_groups * tokens_per_group)


def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    """
    From Huggingface SwitchTransformers

    Examples
    --------
    >>> router_probs = torch.tensor([[[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]]])
    >>> expert_indices = torch.tensor([[2, 1, 0]])
    >>> load_balancing_loss_func(router_probs, expert_indices) >= 1
    True
    """
    num_experts = router_probs.shape[-1]
    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)
    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)
    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)
    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values
    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)
    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts ** 2)

@dataclass
class MoLoRAConfig(LoraConfig):
    """
    Configuration class for LoRA and MoLE.

    Parameters
    ----------
    lora_dim: int
        The dimension of the LoRA layer.
    lora_alpha: int
        The scaling factor for the LoRA layer.
    lora_dropout: float
        The dropout rate for the LoRA layer.
    is_moe: bool
        Whether to use MoLE.
    moe_num_experts: int
        The number of experts in the MoLE layer.
    """
    moe_num_experts: int = field(
        default=8
    )
    router_bias: bool = field(
        default=False
    )
    router_jitter_noise: float = field(
        default=0.01
    )
    router_ignore_padding_tokens: bool = field(
        default=False
    )
    moe_router_z_loss_coef: float = field(
        default=0.001
    )
    moe_router_aux_loss_coef: float = field(
        default=0.1
    )
    # Model parameters
    def __post_init__(self):
        self.peft_type = PeftType.LORA

def get_moepeft_model(model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default"):

    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()

    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

    if peft_config.is_prompt_learning:
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    return PeftModelForCausalLMMoLora(model, peft_config, adapter_name=adapter_name)

def mark_only_molora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
        if "input_layernorm" in n or "post_attention_layernorm" in n:
            p.requires_grad = True
        if "routers" in n:
            p.requires_grad = True
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError

class MoLoRATop1Router(nn.Module):
    """Adapted from Switch Transformers."""

    def __init__(
            self,
            num_experts: int,
            hidden_size: int,
            router_bias: Optional[bool] = False,
            router_jitter_noise: Optional[float] = 0.01,
            router_ignore_padding_tokens: Optional[bool] = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        # self.classifier = nn.Linear(hidden_size, self.num_experts, bias=router_bias)
        self.classifiers = nn.ModuleList(
            [nn.Linear(hidden_size, self.num_experts, bias=router_bias) for _ in range(2)]
        )
        self.jitter_noise = router_jitter_noise
        self.ignore_padding_tokens = router_ignore_padding_tokens

        self.token_inds = None
        # self.expert_capacity =
        # self.dtype = getattr(torch, router_dtype)
        for clfs in self.classifiers:
            clfs.weight.data.normal_(mean=0.0, std=1)

        # self.classifier.weight.data.normal_(mean=0.0, std=1)

    def set_token_inds(self, inds):
        self.token_inds = inds

    def clear_token_inds(self):
        self.token_inds = None

    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # self.input_dtype = hidden_states.dtype
        # hidden_states = hidden_states.to(self.dtype) # debugging

        if self.training and self.jitter_noise > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )

        # Shape: [num_groups, tokens_per_group, num_experts]
        # self._cast_classifier() # debugging
        # fixme: a work around for weird dtype mismatch, even with the casting
        # print(hidden_states.shape)
        origin_shape = hidden_states.shape
        hidden_states = hidden_states.reshape([-1, origin_shape[-1]])
        if self.token_inds is None:
            router_logits = self.classifiers[0](hidden_states)
        else:
            n_routers = int((torch.max(self.token_inds)+1).item())
            router_logits = torch.empty([*hidden_states.shape[:-1], self.num_experts], device=hidden_states.device, dtype=hidden_states.dtype)
            local_token_inds = self.token_inds.reshape(-1)
            for i in range(n_routers):
                # print((local_token_inds == i).sum())
                router_logits[local_token_inds == i] = self.classifiers[i](hidden_states[local_token_inds == i])


        router_logits = router_logits.reshape([*origin_shape[:-1], self.num_experts])
        #     for i in
        # with torch.cuda.amp.autocast(dtype=self.dtype):
        #     router_logits = self.classifier(hidden_states)

        # Apply Softmax and cast back to the original `dtype`
        router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=router_logits.dtype)
        return router_probabilities, router_logits

    # def _cast_classifier(self):
    #     if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
    #         self.classifier = self.classifier.to(self.dtype)

    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        """
        Parameters
        ----------
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
        """
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)

        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # # Mask tokens outside expert capacity. Sum over each sequence
        # token_priority = torch.cumsum(expert_index, dim=-2)
        # # # mask if the token routed to the expert will overflow
        # expert_capacity_mask = token_priority <= self.expert_capacity
        # expert_index = expert_index * expert_capacity_mask

        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)

        return expert_index, router_probs, router_logits

    def add_new_expert_(self, num_new_experts):
        self.num_experts += num_new_experts
        new_classifier = nn.Linear(self.classifier.in_features, self.num_experts,
                                   bias=self.classifier.bias is not None)
        new_classifier.weight.data.normal_(mean=0.0, std=1)
        new_classifier.weight.data[:-num_new_experts, :] = self.classifier.weight.data.clone()
        if self.classifier.bias is not None:
            new_classifier.bias.data[:-num_new_experts] = self.classifier.bias.data.clone()
        self.classifier = new_classifier

class MoLoraLayer(LoraLayer):
    def __init__(self, in_features: int, out_features: int,
                moe_num_experts: Optional[int] = 4,
                router_bias: Optional[bool] = False,
                router_jitter_noise: Optional[float] = 0.01,
                router_ignore_padding_tokens: Optional[bool] = False,
                 **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.moe_num_experts = moe_num_experts
        self.router_bias = router_bias
        self.router_jitter_noise = router_jitter_noise
        self.router_ignore_padding_tokens = router_ignore_padding_tokens
        self.routers = nn.ModuleDict({})

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            for idx in range(self.moe_num_experts):
                nn.init.kaiming_uniform_(self.lora_A[adapter_name][idx].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[adapter_name][idx].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: create_experts(self.in_features, r, self.moe_num_experts)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: create_experts(r, self.out_features, self.moe_num_experts)}))
            self.routers.update(nn.ModuleDict({adapter_name: MoLoRATop1Router(self.moe_num_experts, self.in_features, self.router_bias, self.router_jitter_noise, self.router_ignore_padding_tokens)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

class TaskSpcLoraLayer(LoraLayer):
    def __init__(self, in_features: int, out_features: int,
                moe_num_experts: Optional[int] = 2,
                router_bias: Optional[bool] = False,
                router_jitter_noise: Optional[float] = 0.01,
                router_ignore_padding_tokens: Optional[bool] = False,
                 **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.moe_num_experts = moe_num_experts
        self.router_bias = router_bias
        self.router_jitter_noise = router_jitter_noise
        self.router_ignore_padding_tokens = router_ignore_padding_tokens

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            for idx in range(self.moe_num_experts):
                nn.init.kaiming_uniform_(self.lora_A[adapter_name][idx].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[adapter_name][idx].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: create_experts(self.in_features, r, self.moe_num_experts)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: create_experts(r, self.out_features, self.moe_num_experts)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

class MoLinear(nn.Linear, TaskSpcLoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        moe_num_experts: Optional[int] = 4,
        router_bias: Optional[bool] = False,
        router_jitter_noise: Optional[float] = 0.01,
        router_ignore_padding_tokens: Optional[bool] = False,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        TaskSpcLoraLayer.__init__(self,
                            in_features=in_features,
                            out_features=out_features,
                            moe_num_experts=moe_num_experts,
                            router_bias=router_bias,
                            router_jitter_noise=router_jitter_noise,
                            router_ignore_padding_tokens=router_ignore_padding_tokens)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

        self.last_router_logits = None
        self.last_expert_index = None
        self.idx = 0

    def setting_lora_group(self, idx):
        self.idx = min(idx, self.moe_num_experts-1)

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += self.get_delta_weight(self.active_adapter)
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= self.get_delta_weight(self.active_adapter)
            self.merged = False

    def get_delta_weight(self, adapter):
        all_loras = None
        for idx in range(self.moe_num_experts):
            if all_loras is None:
                all_loras = self.lora_B[adapter][idx].weight @ self.lora_A[adapter][idx].weight
            else:
                all_loras += self.lora_B[adapter][idx].weight @ self.lora_A[adapter][idx].weight
        all_loras /= self.moe_num_experts
        return (
            transpose(
                all_loras,
                self.fan_in_fan_out,
            )
            * self.scaling[adapter]
        )

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            # print(self.lora_A[self.active_adapter][0].weight.dtype)
            x = x.to(self.lora_A[self.active_adapter][0].weight.dtype)
            result += (
                self.lora_B[self.active_adapter][self.idx](
                    self.lora_A[self.active_adapter][self.idx](self.lora_dropout[self.active_adapter](x))
                )
                * self.scaling[self.active_adapter]
            )
            # result += (
            #     self.lora_B[self.active_adapter](
            #         self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
            #     )
            #     * self.scaling[self.active_adapter]
            # )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result

class PeftModelForCausalLMMoLora(PeftModelForCausalLM):
    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        # super().__init__(model, peft_config, adapter_name)
        nn.Module.__init__(self)


        self.base_model = model
        self.config = getattr(self.base_model, "config", {"model_type": "custom"})
        self.modules_to_save = None
        self.peft_config = {}
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        self.peft_config[adapter_name] = peft_config
        self.base_model = MoLoraModel(
            self.base_model, self.peft_config, adapter_name
        )
        self.set_additional_trainable_modules(peft_config, adapter_name)

        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

class MoLoraModel(nn.Module):
    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

        # transformers models have a .config attribute, whose presence is assumed later on
        if not hasattr(self, "config"):
            self.config = {"model_type": "custom"}

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = getattr(self.model, "config", {"model_type": "custom"})
            if hasattr(model_config, "to_dict"):
                model_config = model_config.to_dict()

            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "LoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_molora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit) and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

    def _check_target_module_exists(self, lora_config, key):
        if isinstance(lora_config.target_modules, str):
            target_module_found = re.fullmatch(lora_config.target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            is_using_layer_indexes = getattr(lora_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(lora_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(lora_config.layers_to_transform, int):
                            target_module_found = layer_index == lora_config.layers_to_transform
                        else:
                            target_module_found = layer_index in lora_config.layers_to_transform

                        break
                    else:
                        target_module_found = False
        return target_module_found

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
                target.update_layer_embedding(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )

            elif isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)
            if "ranknum" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        """
        This method merges the LoRa layers into the base model.
        """
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.merge()

    def unmerge_adapter(self):
        """
        This method unmerges the LoRa layers from the base model.
        """
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.unmerge()

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config

    def _unload_and_optionally_merge(self, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def add_weighted_adapter(self, adapters, weights, adapter_name, combination_type="svd"):
        """
        This method adds a new adapter by merging the given adapters with the given weights.

        Args:
            adapters (list): List of adapter names to be merged.
            weights (list): List of weights for each adapter.
            adapter_name (str): Name of the new adapter.
            combination_type (str): Type of merging. Can be one of [`svd`, `linear`]
        """
        if adapter_name in list(self.peft_config.keys()):
            return
        for adapter in adapters:
            if adapter not in list(self.peft_config.keys()):
                raise ValueError(f"Adapter {adapter} does not exist")

        # if there is only one adapter, we can only use linear merging
        combination_type = "linear" if len(adapters) == 1 else combination_type

        # new rank is the max of all ranks of the adapters
        unique_ranks = list({self.peft_config[adapter].r for adapter in adapters})
        if combination_type == "linear":
            if len(unique_ranks) != 1:
                raise ValueError("All adapters must have the same r value when using `linear` combination_type")
            new_rank = unique_ranks[0]
        elif combination_type == "svd":
            new_rank = max(unique_ranks)
        else:
            raise ValueError(f"Invalid combination_type: {combination_type}")

        self.peft_config[adapter_name] = replace(self.peft_config[adapters[0]], r=new_rank, lora_alpha=new_rank)
        self._find_and_replace(adapter_name)
        mark_only_molora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        _freeze_adapter(self.model, adapter_name)
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                if adapter_name in target.lora_A:
                    target_lora_A = target.lora_A[adapter_name].weight
                    target_lora_B = target.lora_B[adapter_name].weight
                elif adapter_name in target.lora_embedding_A:
                    target_lora_A = target.lora_embedding_A[adapter_name]
                    target_lora_B = target.lora_embedding_B[adapter_name]

                target_lora_A.data = target_lora_A.data * 0.0
                target_lora_B.data = target_lora_B.data * 0.0
                if combination_type == "linear":
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.lora_A:
                            current_adapter_lora_A = target.lora_A[adapter].weight
                            current_adapter_lora_B = target.lora_B[adapter].weight
                        elif adapter in target.lora_embedding_A:
                            current_adapter_lora_A = target.lora_embedding_A[adapter]
                            current_adapter_lora_B = target.lora_embedding_B[adapter]
                        target_lora_A.data += current_adapter_lora_A.data * weight * target.scaling[adapter]
                        target_lora_B.data += current_adapter_lora_B.data
                elif combination_type == "svd":
                    target_lora_A.data, target_lora_B.data = self._svd_weighted_adapter(
                        adapters, weights, new_rank, target, target_lora_A, target_lora_B
                    )

    def _svd_weighted_adapter(self, adapters, weights, new_rank, target, target_lora_A, target_lora_B):
        delta_weight = weights[0] * target.get_delta_weight(adapters[0])
        for adapter, weight in zip(adapters[1:], weights[1:]):
            delta_weight += weight * target.get_delta_weight(adapter)
        conv2d = isinstance(target, Conv2d)
        if conv2d:
            conv2d_1x1 = target.weight.size()[2:4] == (1, 1)
            if not conv2d_1x1:
                delta_weight = delta_weight.flatten(start_dim=1)
            else:
                delta_weight = delta_weight.squeeze()
        if target.fan_in_fan_out:
            delta_weight = delta_weight.T

        # based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py#L114-L131
        U, S, Vh = torch.linalg.svd(delta_weight)
        U = U[:, :new_rank]
        S = S[:new_rank]
        U = U @ torch.diag(S)
        Vh = Vh[:new_rank, :]
        dist = torch.cat([U.flatten(), Vh.flatten()])
        hi_val = torch.quantile(dist, 0.99)
        low_val = -hi_val
        U = U.clamp(low_val, hi_val)
        Vh = Vh.clamp(low_val, hi_val)
        if conv2d:
            U = U.reshape(target_lora_B.data.shape)
            Vh = Vh.reshape(target_lora_A.data.shape)
        return Vh, U

    def delete_adapter(self, adapter_name):
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                for attr in [
                    "r",
                    "lora_alpha",
                    "scaling",
                    "lora_A",
                    "lora_B",
                    "lora_embedding_A",
                    "lora_embedding_B",
                    "lora_dropout",
                ]:
                    if adapter_name in getattr(target, attr):
                        getattr(target, attr).pop(adapter_name)
                if target.active_adapter == adapter_name:
                    resetting_active_adapter = list(self.peft_config.keys())[0]
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to {resetting_active_adapter}. "
                    )
                    target.active_adapter = resetting_active_adapter

    def merge_and_unload(self):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge()

    def unload(self):
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)

    def _create_new_module(self, lora_config, adapter_name, target):

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "moe_num_experts": lora_config.moe_num_experts,
            "router_bias": lora_config.router_bias,
            "router_jitter_noise": lora_config.router_jitter_noise,
            "router_ignore_padding_tokens": lora_config.router_ignore_padding_tokens,
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = MoLinear(adapter_name, in_features, out_features, bias=bias, **kwargs)

        return new_module


def _llama_base_model_forward_for_molora(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_router_probs = ()
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
                None,
                use_reentrant=False
            )

        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        # MoLoRA added
        if self.gradient_checkpointing:  # indicating that this is a MoLoRA wrapped layer
            router_probs = layer_outputs[-1]
            all_router_probs = all_router_probs + (router_probs,)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_probs] if v is not None
        )
    return BaseMoEModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        router_logits=all_router_probs,
    )


def _llama_lm_model_forward_for_molora(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    if self.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return MoECausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits if self.is_gradient_checkpointing else None,
    )


def _llava_llama_lm_model_forward_for_molorad(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        proteins: Optional[torch.FloatTensor] = None,
        smiles: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    if inputs_embeds is None:
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            proteins,
            smiles
        )


    return _llama_lm_model_forward_for_molora(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict
    )


# def wrap_transformer_layer_module_for_gradient_checkpointing(
#         model,
#         layer_cls: Union[Type[nn.Module], List[Type[nn.Module]]] = LlamaDecoderLayer,
#         base_model_cls: Type[nn.Module] = LlamaModel,
#         mmlm_model_cls: Type[nn.Module] = LlavaLlamaForCausalLM,
# ):
#     """
#     fixme: only support llama currently
#     Gradient checkpointing will break the computation graph between loss and router classifier,
#     So we need to attach the router logits to the output of each transformer layer.
#     """
#     # if not isinstance(layer_cls, list):
#     #     layer_cls = [layer_cls]
#     # layer_cls = tuple(layer_cls)

#     # def _wrapped_forward(self, *args, **kwargs):
#     #     outputs = self._layer_original_forward(*args, **kwargs)

#     #     layer_router_logits = []
#     #     layer_expert_index = []
#     #     for name, module in self.named_modules():
#     #         if isinstance(module, MoLinear):
#     #             layer_router_logits.append(
#     #                 module.last_router_logits.reshape(-1, *module.last_router_logits.shape[-2:])
#     #             )  # of shape (batch_size, seq_len, num_experts)
#     #             layer_expert_index.append(
#     #                 module.last_expert_index.reshape(-1, module.last_expert_index.shape[-1])
#     #             )  # of shape (batch_size, seq_len)
#     #             # print(module.last_expert_index)
#     #             module.last_router_logits = None
#     #             module.last_expert_index = None

#     #     layer_router_logits = torch.stack(layer_router_logits, dim=-2)
#     #     # of shape (batch_size, seq_len, num_linear, num_experts)
#     #     layer_expert_index = torch.stack(layer_expert_index, dim=-1)
#     #     # of shape (batch_size, seq_len, num_linear)

#     #     router_tuple = (layer_router_logits, layer_expert_index)

#     #     outputs = outputs + (router_tuple,)

#     #     return outputs

#     for name, module in model.named_modules():
#     #     if isinstance(module, layer_cls):
#     #         module._layer_original_forward = module.forward
#     #         module.forward = _wrapped_forward.__get__(module, type(module))
#     #     elif isinstance(module, base_model_cls):
#     #         module._base_model_original_forward = module.forward
#     #         module.forward = _llama_base_model_forward_for_molora.__get__(module, type(module))
#         if isinstance(module, mmlm_model_cls):
#             module._lm_model_original_forward = module.forward
#             module.forward = _llava_llama_lm_model_forward_for_molorad.__get__(module, type(module))

#     return model