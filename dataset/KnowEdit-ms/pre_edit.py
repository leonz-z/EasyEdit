import argparse
import json
from tqdm import tqdm
from typing import Dict, List, Optional, Union
import os
import sys
sys.path.append('../../../EasyEdit')
from easyeditor.editors.utils import _prepare_requests
from easyeditor.evaluate.evaluate import compute_edit_quality
from easyeditor.dataset.knowedit import KnowEditDataset
from easyeditor.models.lora.lora_hparams import LoRAHyperParams
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', type=str, default='Meta-Llama-3-8B-Instruct')
argparser.add_argument('--data_path', type=str, default='../ccks2024_know_edit/ZsRE-test-all.json')
# argparser.add_argument('--device', type=str, default='cuda:0')
argparser.add_argument('--dataset', type=str, default='zsre')
argparser.add_argument('--prefix_prompt', type=int, default=1, help='1 or 3')
args = argparser.parse_args()

model_name = args.model_name
# device=args.device
data_path = args.data_path
dataset = args.dataset
prefix_prompt = args.prefix_prompt
huggingface_cache = os.environ.get('HUGGINGFACE_CACHE')
model_path = huggingface_cache+model_name
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
tok = AutoTokenizer.from_pretrained(model_path)
tok.pad_token_id = tok.eos_token_id

prefix_1 = """Q:What is the native language of Christiane Cohendy?\nA:French\n"""
prefix_3 = """Q:What is the native language of Christiane Cohendy?\nA:French\nQ:What is the final year of Atlanta Flames?\nA:1980\nQ:What is Barbara Legrand's position on the field while playing football?\nA:midfielder\n"""

def get_requests(prompts: Union[str, List[str]],
            target_new: Union[str, List[str]],
            ground_truth: Optional[Union[str, List[str]]] = None,
            rephrase_prompts: Optional[Union[str, List[str]]] = None,
            locality_inputs:  Optional[Dict] = None,
            portability_inputs: Optional[Dict] = None,
            **kwargs):
    if isinstance(prompts, List):
        assert len(prompts) == len(target_new)
    else:
        prompts, target_new = [prompts,], [target_new,]

    if ground_truth is not None:
        ground_truth = [ground_truth,] if isinstance(ground_truth, str) else ground_truth
    else:# Default ground truth is <|endoftext|>
        ground_truth = ['<|endoftext|>'] * (len(prompts))

    requests = _prepare_requests(prompts, target_new, ground_truth, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)
    return requests

def get_data(data_path='../ccks2024_know_edit/ZsRE-test-all.json', prefix_prompt=None):
    datas = KnowEditDataset(data_path)
    if prefix_prompt:
        prompts=[f"{prefix_prompt}Q:{data['prompt']}\nA:" for data in datas]
    else:
        prompts=[data['prompt'] for data in datas]
        
    subjects=[data['subject'] for data in datas]
    target_new = [data['target_new'] for data in datas]

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
    return prompts, subjects, target_new, locality_inputs, portability_inputs


if prefix_prompt == 1:
    prefix = prefix_1
elif prefix_prompt == 3:
    prefix = prefix_3
else:
    raise ValueError("prefix_prompt should be 1 or 3")
prompts, subjects, target_new, locality_inputs, portability_inputs = get_data(data_path, prefix)
request_list = get_requests(
        prompts=prompts,
        target_new=target_new,
        subject=subjects,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
    )


all_metrics = []
hparams = LoRAHyperParams.from_hparams(f'../../hparams/LoRA/{model_name}.yaml')
for i, request in enumerate(tqdm(request_list)):
    metrics = {"pre": compute_edit_quality(model, model_name, hparams, tok, request, hparams.device, eval_metric='exact match', test_generation=True)}
    all_metrics.append(metrics)
json.dump(all_metrics, open(f'{model_name}_{dataset}_{prefix_prompt}.json', 'w'), indent=4)