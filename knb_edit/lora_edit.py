import os
import sys
import json
sys.path.append('../')
import argparse
from easyeditor import LoRAHyperParams
from easyeditor import BaseEditor
from easyeditor import KnowEditDataset

data_dir = '../dataset/ccks2024_know_edit/ZsRE-test-all.json'
train_data_path = None
ds_size, data_type, = 326, 'zsre'
hparams_dir = '../hparams/LoRA/Meta-Llama-3-8B-Instruct'
metrics_save_dir = './EasyEditCache/metrics'

datas = KnowEditDataset(data_dir,size=ds_size)
if data_type == 'counterfact' or data_type == 'recent' or data_type == 'zsre':
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

hparams = LoRAHyperParams.from_hparams(hparams_dir)
pre_file = f"../pre_edit/{hparams.model_name.split('/')[-1]}_{data_type}_pre_edit_{ds_size}.json"
if pre_file is not None and os.path.exists(pre_file):
    pre_edit = json.load(open(pre_file,'r'))
    assert len(pre_edit) == len(prompts)
else:
    pre_edit = None

train_ds = None

editor = BaseEditor.from_hparams(hparams)

# 命令行获取type和p参数
# type取值范围['orgin','abs','square']
# p取值范围 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='orgin')
parser.add_argument('--p', type=float, default=0.5)
args = parser.parse_args()

type_grad, p = args.type, args.p
print(f'0-326-Meta-Llama-3-8B-Instruct-zsre-knb_dict-{type_grad}-{str(p)}')
with open(f'../../knb_dict/0-326-Meta-Llama-3-8B-Instruct-zsre-knb_dict-{type_grad}-{str(p)}.json', 'r') as f:
    knb_dict = json.load(f)
    
# 单条数据编辑
metrics, edited_model, _ = editor.edit(
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
    knb_dict = knb_dict,
)

if not os.path.exists(metrics_save_dir):
    os.makedirs(metrics_save_dir)
json.dump(metrics, open(os.path.join(metrics_save_dir, f'KNB_LoRA_{data_type}_{ds_size}_{hparams_dir.split("/")[-1]}-{type_grad}-{str(p)}-down_proj_results.json'), 'w'), indent=4)