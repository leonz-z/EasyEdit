import os
import os.path
import sys
import json
import random
path =os.getcwd()
print(f'add module root path: {path}')
sys.path.append(path)

from easyeditor.dataset.knowedit import KnowEditDataset
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    PMETHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    GraceHyperParams,
    MENDHyperParams,
    SERACHparams,
    KNBHyperParams,
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import CKnowEditDataset

import argparse

if __name__ == "__main__":
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--editing_method', required=True, type=str)
        parser.add_argument('--data_dir', type=str, default='KE/EasyEdit/dataset/KnowEdit-ms/benchmark_wiki_counterfact_test_cf.json')
        parser.add_argument('--data_type', type=str, default='counterfact', choices=['counterfact', 'zsre', 'wikibio', 'recent'])
        parser.add_argument('--batch_size', default=None, type=int)
        parser.add_argument('--num_steps', default=None, type=int)
        parser.add_argument('--is_post_metrics', default=False, action='store_true')
        parser.add_argument('--p', default=None, type=str)
        parser.add_argument('--knb_dict_path', default=None, type=str)
        parser.add_argument('--t_loss', default=None, type=float)

        parser.add_argument('--metrics_save_dir', default='./knb_output/', type=str)
        parser.add_argument('--hparams_dir', type=str, default='./hparams/KNB/Llama-2-7b-ms.yaml')
        parser.add_argument('--pre_file', default='./pre_edit/Llama-2-7b-hf_counterfact_pre_edit_all_v2.json', type=str)
        parser.add_argument('--layers', default=None, type=str)
        parser.add_argument('--target_modules', default=None, type=str) 
        # 目标模块，all,mlp,attn,需要和knb_dict的weighs name对应
        parser.add_argument('--start_idx_end_idx', default=None, type=str)
        # 基于lora实现的knb方法
        parser.add_argument('--lora_type', default=None, type=str, help='lora type: lora,adalora')
        parser.add_argument('--ds_size', default=None, type=int)
        args = parser.parse_args()
        return args
    args = get_args()
    def get_editing_hparams():
        if args.editing_method == 'FT':
            editing_hparams = FTHyperParams
        elif args.editing_method == 'IKE':
            editing_hparams = IKEHyperParams
        elif args.editing_method == 'KN':
            editing_hparams = KNHyperParams
        elif args.editing_method == 'MEMIT':
            editing_hparams = MEMITHyperParams
        elif args.editing_method == 'PMET':
            editing_hparams = PMETHyperParams
        elif args.editing_method == 'ROME':
            editing_hparams = ROMEHyperParams
        elif args.editing_method == 'LoRA':
            editing_hparams = LoRAHyperParams
        elif args.editing_method == 'GRACE':
            editing_hparams = GraceHyperParams
        elif args.editing_method == 'MEND':
            editing_hparams = MENDHyperParams
        elif args.editing_method == 'KNB':
            editing_hparams = KNBHyperParams
        else:
            raise NotImplementedError
        return editing_hparams
    def get_data():
        # 加载处理数据
        datas = KnowEditDataset(args.data_dir,size=args.ds_size)
        if args.start_idx_end_idx is not None:
            start_idx, end_idx = args.start_idx_end_idx.split(',')
            datas = datas[int(start_idx):int(end_idx)]
        if args.datatype == 'counterfact' or args.datatype == 'recent' or args.datatype == 'zsre':
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
        if args.datatype == 'wikibio':
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

        return prompts, target_new, ground_truth, subjects, rephrase_prompts, locality_inputs, portability_inputs
    prompts, target_new, ground_truth, subject, rephrase_prompts, locality_inputs, portability_inputs = get_data()
    # 处理参数
    if args.pre_file is not None and os.path.exists(args.pre_file):
        pre_edit = json.load(open(args.pre_file,'r', encoding='utf-8'))
        if args.start_idx_end_idx is not None:
            start_idx, end_idx = args.start_idx_end_idx.split(',')
            pre_edit = pre_edit[int(start_idx):int(end_idx)]
        assert len(pre_edit) == len(prompts)
    else:
        pre_edit = None      
    editing_hparams = get_editing_hparams()
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    if args.target_modules is not None:
        # LORA
        if args.target_modules == 'all':
            pass
        elif args.target_modules in ['mlp', 'attn']:
            target_modules = [m for m in hparams.target_modules if args.target_modules in m]
            hparams.target_modules = target_modules
        # KNB
        else:
            hparams.target_modules = args.target_modules.split(',')
        print(f"target_modules: {hparams.target_modules}")
    if args.layers is not None:
        start_layer, end_layer = args.layers.split(',')
        layers = [i for i in range(int(start_layer), int(end_layer))]
        hparams.layers = layers
        print(f"layers: {hparams.layers}")
    if args.num_steps is not None:
        hparams.num_steps = args.num_steps
    if args.batch_size is not None:
        hparams.batch_size = args.batch_size
    if args.lora_type is not None:
        hparams.lora_type = args.lora_type
    if args.t_loss is not None:
        hparams.t_loss = args.t_loss
    if args.max_new_tokens_times is not None:
        max_new_tokens_times = args.max_new_tokens_times
    else:
        max_new_tokens_times = 1
    # 保存文件名
    save_name = f'{args.data_type}_{args.editing_method}_{hparams.model_name.split("/")[-1]}'
    if args.layers is not None:
        save_name = f'{save_name}_{args.layers}'
    elif hasattr(hparams, 'layers') and len(hparams.layers) > 0:
        save_name = f'{save_name}_{hparams.layers[0]}_{hparams.layers[-1]}'
    if args.num_steps is not None:
        save_name = f'{save_name}_{args.num_steps}'
    if args.batch_size is not None:
        save_name = f'{save_name}_{args.batch_size}'
    if args.target_modules is not None:
        save_name = f'{save_name}_{args.target_modules}'
    elif hasattr(hparams, 'target_modules') and len(hparams.target_modules) > 0:
        save_name = f'{save_name}_{"_".join(hparams.target_modules)}'
    if args.start_idx_end_idx is not None:
        save_name = f'{args.start_idx_end_idx}_{save_name}'

    knb_dict_list = None
    if args.editing_method == 'LoRA':
        save_name = f'{save_name}_r{hparams.rank}_p{hparams.lora_dropout}'
        save_name = f'{save_name}_rs{hparams.use_rslora}_a{hparams.lora_alpha}'
        save_name = f'{save_name}_b_{hparams.bias}_tr{hparams.target_r}_ir{hparams.init_r}'
    elif args.editing_method == 'KNB':
        hparams.p = args.p
        # save_name = f"{args.start_idx_end_idx}_{args.knb_dict_path.split('/')[-1].replace('.json', '')}"
        save_name = f"{args.knb_dict_path.split('/')[-1].replace('.json', '')}"
        save_name += f'_{args.num_steps}_bs{hparams.batch_size}'
        save_name = f'{save_name}_rs{hparams.use_rsknb}_a{hparams.knb_alpha}'
        save_name = f'{save_name}_pd{hparams.knb_dropout}_bias_{hparams.bias}_t_loss{hparams.t_loss}'
        save_name = f'{save_name}_wd{hparams.weight_decay}_tt{max_new_tokens_times}'
        with open(args.knb_dict_path, 'r', encoding='utf-8') as f:
            knb_dict_list = json.load(f)
    print(f"Hparams:\n{save_name}")
    print(hparams)
    # 保存结果
    if not os.path.exists(args.metrics_save_dir+'/log'):
        os.makedirs(args.metrics_save_dir+'/log')
    if not os.path.exists(args.metrics_save_dir+'/result'):
        os.makedirs(args.metrics_save_dir+'/result')
    # 编辑模型
    editor = BaseEditor.from_hparams(hparams)
    if hparams.batch_size == 1:
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new=target_new,
            subject=subject,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            keep_original_weight=True,
            pre_edit=pre_edit,
            test_generation=True,
            knb_dict_list=knb_dict_list,
            file_obj = open(os.path.join(args.metrics_save_dir, f'log/{save_name}_log.json'), encoding='utf-8', mode='a'),
        )
    else:
        metrics, edited_model, _ = editor.batch_edit(
            prompts=prompts,
            target_new=target_new,
            subject=subject,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            keep_original_weight=True,
            pre_edit=pre_edit,
            test_generation=True,
            knb_dict_list=knb_dict_list,
            file_obj = open(os.path.join(args.metrics_save_dir, f'log/{save_name}_log.json'), encoding='utf-8', mode='a'),
        )
    
    json.dump(metrics,
            open(os.path.join(args.metrics_save_dir, f'result/{save_name}.json'), encoding='utf-8', mode='w'),
            indent=4,
            ensure_ascii=False)