import json
from dataclasses import dataclass
from dataclasses import asdict


@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """
    # cjc@0529 @dataclass 装饰器,封装数据
    # cjc@0530 fix bug: TypeError: non-default argument 'model_name' follows default argument
    # cjc@0530 TypeError: __init__() missing 2 required positional arguments: 'fp16' and 'bf16'
    # fp16: bool
    # bf16: bool
    
    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)

        return cls(**data)

    def construct_float_from_scientific_notation(config: dict):
        for key, value in config.items():
            if isinstance(value, str):
                try:
                    # Convert scalar to float if it is in scientific notation format
                    config[key] = float(value)
                except:
                    pass
        return config
    
    def to_dict(config) -> dict:
        dict = asdict(config)
        return dict
            
        

    # @classmethod
    # def from_hparams(cls, hparams_name_or_path: str):
    #
    #     if '.yaml' not in hparams_name_or_path:
    #         hparams_name_or_path = hparams_name_or_path + '.yaml'
    #     config = compose(hparams_name_or_path)
    #
    #     assert config.alg_name in ALG_DICT.keys() or print(f'Editing Alg name {config.alg_name} not supported yet.')
    #
    #     params_class, apply_algo = ALG_DICT[config.alg_name]
    #
    #     return params_class(**config)
