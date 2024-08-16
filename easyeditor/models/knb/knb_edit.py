# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import logging
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
module_path = '/home/bingxing2/home/scx7avs/lyc/EasyEdit'
print(f'add {module_path} to sys path')
sys.path.append(module_path)
# from easyeditor.models.lora.peft import get_peft_model, TaskType, KnbConfig
from .peft import get_peft_model, TaskType, KnbConfig
# from easyeditor.evaluate.evaluate import compute_edit_quality
# from easyeditor.util import nethook
from .utils import nethook
from .utils.evaluate import compute_edit_quality

# %% [markdown]
# 配置文件
# %% [markdown]
# log日志

# %%
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


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)

def make_logs():
    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)

make_logs()

# %% [markdown]
# knb训练

# %%
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
def apply_knb_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: KNBHyperParams, # type: ignore
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

    edited_model = execute_knb(model, tok, requests, hparams, keep_original_weight, **kwargs)

    return edited_model, weights_copy

@gpu_mem_report
def knb_forward(peft_model, txt, tgt, mask_token, device, tok, loss_meter, opt):
    full_prompt = [f"{p} {l}" for p, l in zip(txt, tgt)]
    tok.pad_token = tok.eos_token
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
def execute_knb(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: KNBHyperParams, # type: ignore
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

    if not keep_original_weight and hasattr(model,'peft_config'):
        peft_model = model
    else:
        if kwargs.get('knb_dict'):
            knb_dict = kwargs['knb_dict']
        else:
            knb_dict = None
            print("No knb_dict provided")
        peft_config = KnbConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            knb_alpha=hparams.knb_alpha, knb_dropout=hparams.knb_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules, # target_knb
            knb_dict=knb_dict,
            use_rsknb=True,
        )
        peft_model = get_peft_model(model, peft_config)

    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    # peft_model.print_trainable_parameters()
    requests = deepcopy(requests) # 训练log中观察到每次request都是一个batch_size大小
    for request in requests:
        print(
            f"Executing KNB algo for: "
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
            knb_forward(peft_model, txt, tgt, mask_token, device, tok, loss_meter, opt)
        
        print(f"Total loss {loss_meter.avg}")

        # if loss_meter.avg < 1e-3:
        #     break
    return peft_model

# %%
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

# %% [markdown]
# 批量编辑

# %%
def batch_edit(hparams: HyperParams,
                model: AutoModelForCausalLM,
                tok: AutoTokenizer,
                prompts: List[str],
                target_new: List[str],
                ground_truth: Optional[List[str]] = None,
                rephrase_prompts: Optional[List[str]] = None,
                locality_inputs:  Optional[Dict] = None,
                portability_inputs: Optional[Dict] = None,
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
            edited_model, weights_copy = apply_knb_to_model(
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
            return None, None, None
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

# %% [markdown]
# 加载数据

# %%
from easyeditor import KnowEditDataset

model_name = 'Llama-2-7b-hf'
type_grad, p = 'max', 99.85
data_type = 'counterfact'
ds_size = 'all'
data_dir = '../dataset/KnowEdit-ms/benchmark_wiki_counterfact_test_cf.json'
train_data_path = None
no_prompts = True
print(model_name)
hparams_dir = f'../hparams/KNB/{model_name}'
print(f"hparams_dir: {hparams_dir}")
metrics_save_dir = f'./EasyEditCache/metrics/{ds_size}-{data_type}-knb/'

size =  None if ds_size=='all' else int(ds_size)
datas = KnowEditDataset(data_dir, size=size)
if data_type == 'counterfact' or data_type == 'recent' or data_type == 'zsre':
    # f"Please answer the question in no more than {answer_len} words!\nQuestion:{query}\nAnswer:"
    if not no_prompts:
        prompts, subjects, target_new = [], [], []
        for data in datas:
            subjects.append(data['subject'])
            target_new.append(data['target_new'])
            answer_len = len(data['target_new'].split(' '))
            prompts.append(f"Please answer the question in no more than {answer_len} words!\nQuestion:{data['prompt']}\nAnswer:")
    else:
        prompts=[data['prompt'] for data in datas]
        subjects=[data['subject'] for data in datas]
        target_new = [data['target_new'] for data in datas]
    
    ground_truth = [data['ground_truth'] for data in datas]
    portability_r =[data['portability_r'] for data in datas]
    portability_s =[data['portability_s'] for data in datas]
    portability_l =[data['portability_l'] for data in datas]

    portability_reasoning_prompts=[]
    portability_reasoning_ans=[]
    portability_Logical_Generalization_prompts=[]
    portability_Logical_Generalization_ans=[]
    portability_Subject_Aliasing_prompts=[]
    portability_Subject_Aliasing_ans=[]
    
    portability_data = [portability_r,portability_s,portability_l]
    portability_prompts = [portability_reasoning_prompts,portability_Subject_Aliasing_prompts,portability_Logical_Generalization_prompts]
    portability_answers = [portability_reasoning_ans,portability_Subject_Aliasing_ans,portability_Logical_Generalization_ans]
    for data, portable_prompts, portable_answers in zip(portability_data,portability_prompts,portability_answers):
        for item in data:
            if item is None:
                portable_prompts.append(None)
                portable_answers.append(None)
            else:
                temp_prompts = []
                temp_answers = []
                for pr in item:
                    prompt=pr["prompt"]
                    an=pr["ground_truth"]
                    while isinstance(an,list):
                        if an==[]:
                            an=''
                        else:
                            an = an[0]
                    if an.strip() =="":
                        continue
                    temp_prompts.append(prompt)
                    temp_answers.append(an)
                portable_prompts.append(temp_prompts)
                portable_answers.append(temp_answers)
    assert len(prompts) == len(portability_reasoning_prompts) == len(portability_Logical_Generalization_prompts) == len(portability_Subject_Aliasing_prompts)
    
    locality_rs = [data['locality_rs'] for data in datas]
    locality_f = [data['locality_f'] for data in datas]
    locality_Relation_Specificity_prompts=[]
    locality_Relation_Specificity_ans=[]
    locality_Forgetfulness_prompts=[]        
    locality_Forgetfulness_ans=[]
    
    locality_data = [locality_rs, locality_f]
    locality_prompts = [locality_Relation_Specificity_prompts,locality_Forgetfulness_prompts]
    locality_answers = [locality_Relation_Specificity_ans,locality_Forgetfulness_ans]
    for data, local_prompts, local_answers in zip(locality_data,locality_prompts,locality_answers):
        for item in data:
            if item is None:
                local_prompts.append(None)
                local_answers.append(None)
            else:
                temp_prompts = []
                temp_answers = []
                for pr in item:
                    prompt=pr["prompt"]
                    an=pr["ground_truth"]
                    while isinstance(an,list):
                        if an==[]:
                            an=''
                        else:
                            an = an[0]
                    if an.strip() =="":
                        continue
                    temp_prompts.append(prompt)
                    temp_answers.append(an)
                local_prompts.append(temp_prompts)
                local_answers.append(temp_answers)
    assert len(prompts) == len(locality_Relation_Specificity_prompts) == len(locality_Forgetfulness_prompts)
    locality_inputs = {}
    portability_inputs = {}
    
    locality_inputs = {
        'Relation_Specificity':{
            'prompt': locality_Relation_Specificity_prompts,
            'ground_truth': locality_Relation_Specificity_ans
        },
        'Forgetfulness':{
            'prompt':locality_Forgetfulness_prompts,
            'ground_truth':locality_Forgetfulness_ans
        }
    }
    portability_inputs = {
        'Subject_Aliasing':{
            'prompt': portability_Subject_Aliasing_prompts,
            'ground_truth': portability_Subject_Aliasing_ans
        },
        'reasoning':{
            'prompt': portability_reasoning_prompts,
            'ground_truth': portability_reasoning_ans           
        },
        'Logical_Generalization':{
            'prompt': portability_Logical_Generalization_prompts,
            'ground_truth': portability_Logical_Generalization_ans           
        }
    }
    
if data_type == 'wikibio':
    prompts=[data['prompt'] for data in datas]
    subjects=[data['subject'] for data in datas]
    target_new = [data['target_new'] for data in datas]
    
    locality_rs = [data['locality_rs'] for data in datas]
    locality_f = [data['locality_f'] for data in datas]
    locality_Relation_Specificity_prompts=[]
    locality_Relation_Specificity_ans=[]
    
    locality_data = [locality_rs]
    locality_prompts = [locality_Relation_Specificity_prompts]
    locality_answers = [locality_Relation_Specificity_ans]
    for data, local_prompts, local_answers in zip(locality_data,locality_prompts,locality_answers):
        for item in data:
            if item is None:
                local_prompts.append(None)
                local_answers.append(None)
            else:
                temp_prompts = []
                temp_answers = []
                for pr in item:
                    prompt=pr["prompt"]
                    an=pr["ground_truth"]
                    while isinstance(an,list):
                        if an==[]:
                            an=''
                        else:
                            an = an[0]
                    if an.strip() =="":
                        continue
                    temp_prompts.append(prompt)
                    temp_answers.append(an)
                local_prompts.append(temp_prompts)
                local_answers.append(temp_answers)
    assert len(prompts) == len(locality_Relation_Specificity_prompts)
    portability_inputs = None
    locality_inputs = {}
    locality_inputs = {
        'Relation_Specificity':{
            'prompt': locality_Relation_Specificity_prompts,
            'ground_truth': locality_Relation_Specificity_ans
        }
    }

# %% [markdown]
# 加载模型

# %%
path = '/home/bingxing2/public/models/llama2/'
model = AutoModelForCausalLM.from_pretrained(path+model_name, device_map='auto', torch_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(path+model_name)

# %% [markdown]
# 读取配置文件和设置参数

# %%
hparams = KNBHyperParams.from_hparams(hparams_dir)
hparams.batch_size = 60
hparams.num_steps = 50
if ds_size is not None:
    pre_file = f"../pre_edit/{hparams.model_name}_{data_type}_pre_edit_{ds_size}.json"
else:
    pre_file = f"../pre_edit/{hparams.model_name}_{data_type}_pre_edit.json"
if pre_file is not None and os.path.exists(pre_file):
    pre_edit = json.load(open(pre_file,'r'))
    assert len(pre_edit) == len(prompts)
else:
    pre_edit = None

train_ds = None
with open(f'../../knb_dict/all-llama2-{data_type}/all-Llama-2-7b-hf-{data_type}-knb_dict-orgin-{type_grad}-{p}.json', 'r') as f:
    knb_dict = json.load(f)

# %%
knb_dict_new = {
    'mlp.down_proj': knb_dict
}

# %% [markdown]
# 执行编辑

# %%
metrics, edited_model, _ = batch_edit(
    hparams=hparams,
    model=model,
    tok=tok,
    prompts=prompts,
    target_new=target_new,
    subject=subjects,
    locality_inputs=locality_inputs,
    portability_inputs=portability_inputs,
    train_ds=train_ds,
    keep_original_weight=True,
    pre_file=pre_file,
    pre_edit = pre_edit,
    test_generation=True,
    knb_dict = knb_dict_new,
)    

# %%
if not os.path.exists(metrics_save_dir):
    os.makedirs(metrics_save_dir)
    
if hparams.no_prompts:
    json.dump(metrics, \
            open(metrics_save_dir + \
                f'KNB_{hparams.alg_name}_{data_type}_{ds_size}_{hparams.model_name}_{type_grad}_{p}_{hparams.batch_size}_{hparams.num_steps}_{"_".join(hparams.target_modules)}_no_prompts.json', 'w'), indent=4)
else:
    json.dump(metrics, \
            open(metrics_save_dir + \
                f'KNB_{hparams.alg_name}_{data_type}_{ds_size}_{hparams.model_name}_{type_grad}_{p}_{hparams.batch_size}_{hparams.num_steps}_{"_".join(hparams.target_modules)}.json', 'w'), indent=4)


