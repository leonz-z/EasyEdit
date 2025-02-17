from dataclasses import dataclass
from typing import List
from ...util.hparams import HyperParams
import yaml


@dataclass
class LoRAHyperParams(HyperParams):
    # Method
    lora_type: str
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float
    target_modules: List[str]
    rank: int
    lora_alpha: float
    lora_dropout: float
    # Module templates

    device: int
    alg_name: str
    model_name: str

    # Defaults
    batch_size: int = 128
    max_length: int = 40
    model_parallel: bool = False

    bf16: bool = False
    fp16: bool = False
    # lora
    use_dora: bool = False
    use_rslora: bool = False
    bias: str = 'none'
    # adalora
    target_r: int = 8
    init_r: int = 16
    deltaT: int = 1
    beta1: float = 0.85
    beta2: float = 0.85
    orth_reg_weight: float = 0.5

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'LoRA') or print(
            f'LoRAHyperParams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} ')
        return cls(**config)
