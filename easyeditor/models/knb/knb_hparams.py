from dataclasses import dataclass
from typing import List
from ...util.hparams import HyperParams
import yaml


    
@dataclass
class KNBHyperParams(HyperParams):
    # Method
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float
    target_modules: List[str]
    knb_alpha: float
    knb_dropout: float
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
    
    use_rsknb: bool = False
    bias: str = 'none'
    p: str = None
    t_loss: float = 1e-3
    knb_layer: str = None # this_layer, last_layer

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'KNB') or print(
            f'KNBHyperParams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} ')
        return cls(**config)
