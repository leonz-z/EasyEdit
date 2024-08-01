import logging
import os
import json
import time
import yaml
import torch
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore
from dataclasses import dataclass, asdict
import sys
module_path = '/home/lyc/TNTprojectz/KE/EasyEdit'
print(f'add {module_path} to sys path')
sys.path.append(module_path)
from easyeditor.models.lora.peft import get_peft_model, AdaLoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig
from easyeditor.evaluate.evaluate import compute_edit_quality

@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """
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
    
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk

def gpu_mem_report(func):
    # gpu_mem=24
    def wrapper(*args, **kwargs):
        # mem_used = torch.cuda.memory_allocated() / 1024 ** 3
        # print(f"before {func.__name__}: {mem_used:.2f} GB {mem_used/gpu_mem*100:.2f}%")
        res = func(*args, **kwargs)
        # mem_used = torch.cuda.memory_allocated() / 1024 ** 3
        # print(f"after {func.__name__}: {mem_used:.2f} GB {mem_used/gpu_mem*100:.2f}%")
        # print(f"by PyTorch: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB {torch.cuda.memory_allocated() / 1024 ** 3 / gpu_mem * 100:.2f}%")
        torch.cuda.empty_cache()
        # mem_used = torch.cuda.memory_allocated() / 1024 ** 3
        # print(f"after empty cache: {mem_used:.2f} GB {mem_used/gpu_mem*100:.2f}%")
        # print(f"by PyTorch: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB {torch.cuda.memory_allocated() / 1024 ** 3 / gpu_mem * 100:.2f}%")
        
        return res
    
    return wrapper

@gpu_mem_report
def apply_lora_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: LoRAHyperParams, # type: ignore
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    edited_model = execute_lora(model, tok, requests, hparams, keep_original_weight, **kwargs)

    return edited_model, weights_copy

@gpu_mem_report
def lora_forward(peft_model, txt, tgt, mask_token, device, tok, loss_meter, opt):
    full_prompt = [f"{p} {l}" for p, l in zip(txt, tgt)]
    prompt_ids = tok(list(txt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_ids]
    tokens = tok(full_prompt, return_tensors="pt", padding=True, truncation=True)
    bs = tokens["input_ids"].shape[0]
    tokens["labels"] = tokens["input_ids"].clone()
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in tokens["labels"]]
    for i in range(len(txt)):
        tokens["labels"][i][num_pad_toks[i]:num_pad_toks[i]+num_prompt_toks[i]] = mask_token
    tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = mask_token
    tokens = tokens.to(device)
    pred = peft_model(**tokens)
    loss = pred.loss
    print(f"Batch loss {loss.item()}")
    loss_meter.update(loss.item(), n=bs)
    # if loss.item() >= 1e-3:
    loss.backward()
    opt.step()

@gpu_mem_report
def execute_lora(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: LoRAHyperParams, # type: ignore
        keep_original_weight=False,
        **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the Lora update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    model.config.use_cache = False
    model.supports_gradient_checkpointing = True  #
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if hparams.lora_type == "lora":
        Config = LoraConfig
    elif hparams.lora_type == "adalora":
        Config = AdaLoraConfig
    else:
        raise NotImplementedError
    if not keep_original_weight and hasattr(model,'peft_config'):
        peft_model = model
    else:
        if kwargs.get('knb_dict'):
            knb_dict = kwargs['knb_dict']
        else:
            knb_dict = None
            print("No knb_dict provided")
        peft_config = Config(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hparams.rank,
            lora_alpha=hparams.lora_alpha, lora_dropout=hparams.lora_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules, # target_knb
            knb_dict=knb_dict,
        )
        peft_model = get_peft_model(model, peft_config)

    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    # peft_model.print_trainable_parameters()
    requests = deepcopy(requests) # 训练log中观察到每次request都是一个batch_size大小
    for request in requests:
        print(
            f"Executing LoRA algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    device = torch.device(f'cuda:{hparams.device}')
    print(f"Using device: {device}")
    # Define inputs
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]

    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        peft_model.parameters(),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt, tgt in zip( # 输入数据为batch_size条,只循环一次
                chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            mask_token = -100
            opt.zero_grad()
            lora_forward(peft_model, txt, tgt, mask_token, device, tok, loss_meter, opt)
        
        print(f"Total loss {loss_meter.avg}")

        # if loss_meter.avg < 1e-3:
        #     break
    return peft_model

def get_handler(path, log_name):
    log_file_path = os.path.join(path, log_name)
    try:
        if not os.path.exists(path):
            print("We are creating the logger files")
            os.makedirs(path)
    except:
        pass
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    return file_handler, stream_handler

def _prepare_requests(prompts: Union[str, List[str]],
                      target_new: Union[str, List[str]],
                      ground_truth: Union[str, List[str]],
                      rephrase_prompts: Optional[Union[str, List[str]]] = None,
                      locality_inputs: Optional[Dict] = None,
                      portability_inputs: Optional[Dict] = None,
                      **kwargs
                      ):

    requests = [{
        'prompt': prompt,
        'target_new': target_new_,
        'ground_truth': ground_truth_,
        'portability': {},
        'locality': {}
    }
    for prompt, ground_truth_, target_new_ in zip(prompts, ground_truth, target_new)
    ]

    if 'subject' in kwargs:
        if isinstance(kwargs['subject'], str):
            kwargs['subject'] = [kwargs['subject'],]
        else:
            assert len(kwargs['subject']) == len(prompts)
        for prompt_, subject_ in zip(prompts, kwargs['subject']):
            assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

        for i, request in enumerate(requests):
            request.update(
                {
                    'subject': kwargs['subject'][i]
                }
            )
    if 'loc_prompts' in kwargs:
        if isinstance(kwargs['loc_prompts'], str):
            kwargs['loc_prompts'] = [kwargs['loc_prompts'],]
        else:
            assert len(kwargs['loc_prompts']) == len(prompts)

        for i, request in enumerate(requests):
            request.update(
                {
                    'loc_prompt': kwargs['loc_prompts'][i]
                }
            )

    if rephrase_prompts is not None:
        if isinstance(rephrase_prompts, str):
            rephrase_prompts = [rephrase_prompts,]

        for i, request in enumerate(requests):
            request.update(
                {
                    'rephrase_prompt': rephrase_prompts[i],
                }
            )
    if locality_inputs is not None:
        for locality_key in locality_inputs.keys():
            if isinstance(locality_inputs[locality_key]['prompt'], str):
                locality_inputs[locality_key]['prompt'] = [locality_inputs[locality_key]['prompt'],]
                locality_inputs[locality_key]['ground_truth'] = [locality_inputs[locality_key]['ground_truth'], ]
            assert len(locality_inputs[locality_key]['prompt']) == len(locality_inputs[locality_key]['ground_truth']) \
            == len(requests), print('One Edit instance needs one locality input.....')

            for i, request in enumerate(requests):
                if locality_inputs[locality_key]['prompt'][i] is not None:
                    request['locality'].update(
                        {
                            locality_key: {
                                f'prompt': locality_inputs[locality_key]['prompt'][i],
                                f'ground_truth': locality_inputs[locality_key]['ground_truth'][i]
                            }
                        }
                    )

    if portability_inputs is not None:
        for portability_key in portability_inputs.keys():
            if isinstance(portability_inputs[portability_key]['prompt'], str):
                portability_inputs[portability_key]['prompt'] = [portability_inputs[portability_key]['prompt'],]
                portability_inputs[portability_key]['ground_truth'] = [portability_inputs[portability_key]['ground_truth'], ]
            assert len(portability_inputs[portability_key]['prompt']) == len(portability_inputs[portability_key]['ground_truth']) \
            == len(requests), 'One Edit instance needs one portability input.....'

            for i, request in enumerate(requests):
                if portability_inputs[portability_key]['prompt'][i] is not None:
                    request['portability'].update(
                        {
                            portability_key: {
                                'prompt': portability_inputs[portability_key]['prompt'][i],
                                'ground_truth': portability_inputs[portability_key]['ground_truth'][i]
                            }
                        }
                    )
    return requests

def _chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i: i + n]

from easyeditor.util import nethook


def batch_edit(hparams: HyperParams,
                model: AutoModelForCausalLM,
                tok: AutoTokenizer,
                prompts: List[str],
                target_new: List[str],
                ground_truth: Optional[List[str]] = None,
                rephrase_prompts: Optional[List[str]] = None,
                locality_inputs:  Optional[Dict] = None,
                portability_inputs: Optional[Dict] = None,
                keep_original_weight=False,
                verbose=True,
                **kwargs
                ):
    """
    `prompts`: list or str
        the prompts to edit
    `ground_truth`: str
        the ground truth / expected output
    """
    assert len(prompts) == len(target_new)
    test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
    if ground_truth is not None:
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth,]
        else:
            assert len(ground_truth) == len(prompts)
    else: # Default ground truth is <|endoftext|>
        ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


    # 2024-7-13 locality_inputs portability_inputs
    requests = _prepare_requests(prompts, target_new, ground_truth, rephrase_prompts,
                                        locality_inputs, portability_inputs, **kwargs)
    torch.cuda.empty_cache()
    assert hasattr(hparams, 'batch_size'), f'Method {hparams.alg_name} found, pls specify the batch_size....'
    all_metrics = []
    for record_chunks in _chunks(requests, hparams.batch_size):
        start = time.time()
        if kwargs.get('knb_dict'):
            knb_dict = kwargs['knb_dict']
            edited_model, weights_copy = apply_lora_to_model(
                model,
                tok,
                record_chunks,
                hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=False,
                knb_dict=knb_dict
            )
            torch.cuda.empty_cache()
        else:
            print('no knb_dict, use default LoRA')
            edited_model, weights_copy = apply_lora_to_model(
                model,
                tok,
                record_chunks,
                hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=False,
            )
        exec_time = time.time() - start
        LOG.info(f"Execution editing took {exec_time}")

        start = time.time()
        chunk_metrics = []
        for i, request in enumerate(record_chunks):

            metrics = {
                'case_id': i,
                "requested_rewrite": request,
                "time": exec_time,
                "post": compute_edit_quality(edited_model, hparams.model_name, hparams, tok, request, hparams.device, test_generation=test_generation),
            }
            torch.cuda.empty_cache()
            chunk_metrics.append(metrics)

        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to(f"cuda:{hparams.device}")
        torch.cuda.empty_cache()
        for i, request in enumerate(record_chunks):
            chunk_metrics[i]["pre"] = compute_edit_quality(model, hparams.model_name, hparams, tok, request, hparams.device, test_generation=test_generation)
            torch.cuda.empty_cache()
            if verbose:
                LOG.info(
                    f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {chunk_metrics[i]}"
                )
        
        LOG.info(f"Evaluation took {time.time() - start}")
        all_metrics.extend(chunk_metrics)
    return all_metrics, edited_model, weights_copy