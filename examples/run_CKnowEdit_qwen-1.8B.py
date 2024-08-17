import os
import os.path
import sys
import json
import random
sys.path.append('/home/lyc/TNTprojectz/KE/EasyEdit')
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
        parser.add_argument('--data_dir', type=str, default='./dataset/ccks2024_know_edit/ccks-CKnowEdit.json')
        parser.add_argument('--metrics_save_dir', default='./ccks2024_output/', type=str)
        parser.add_argument('--batch_size', default=None, type=int)
        parser.add_argument('--num_steps', default=None, type=int)
        parser.add_argument('--is_post_metrics', default=False, action='store_true')
        parser.add_argument('--p', default=None, type=str)
        parser.add_argument('--knb_dict_path', default=None, type=str)
        
        parser.add_argument('--hparams_dir', type=str, default='./hparams/LoRA/Qwen-1_8B-Chat.yaml')
        parser.add_argument('--ds_size', default=None, type=int)
        parser.add_argument('--train_data_path', type=str, default='./dataset/ccks2024_know_edit/ccks-CKnowEdit.json')
        parser.add_argument('--pre_file', default='./pre_edit/Qwen-1_8B-Chat_CKnowEdit_pre_edit.json', type=str)
        parser.add_argument('--data_type', type=str, default='CKnowEdit')
        parser.add_argument('--layers', default=None, type=str)
        parser.add_argument('--target_modules', default=None, type=str) # 目标模块，all,mlp,attn
        parser.add_argument('--start_idx_end_idx', default=None, type=str)
        parser.add_argument('--lora_type', default=None, type=str, help='lora type: lora,adalora')
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
        datas = CKnowEditDataset(args.data_dir,size=args.ds_size)
        # datas = datas[477:480] # debug
        # datas = datas[424:424+1] # debug
        # datas = datas[:424] # IKE debug
        if args.start_idx_end_idx is not None:
            start_idx, end_idx = args.start_idx_end_idx.split(',')
            datas = datas[int(start_idx):int(end_idx)]
        prompts=[data['prompt'] for data in datas]
        target_new = [data['target_new'] for data in datas]
        ground_truth = [data['target_old'] for data in datas]
        subject = [data['subject'] for data in datas]
        rephrase_prompts = [data['rephrase'] for data in datas]
        portability_data =[data['portability'] for data in datas]
        locality_data = [data['locality'] for data in datas]

        portability_prompts=[]
        portability_answers=[]
        for item in portability_data:
            if item is None:
                portability_prompts.append(None)
                portability_answers.append(None)
            else:
                temp_prompts = []
                temp_answers = []
                for pr in item:
                    prompt=pr['prompt']
                    an=pr['answer']
                    temp_prompts.append(prompt)
                    temp_answers.append(an)
                portability_prompts.append(temp_prompts)
                portability_answers.append(temp_answers)
        assert len(prompts)==len(portability_prompts)==len(portability_answers)

        locality_prompts=[]
        locality_answers=[]
        for item in locality_data:
            if item is None:
                locality_prompts.append(None)
                locality_answers.append(None)
            else:
                temp_prompts = []
                temp_answers = []
                for pr in item:
                    if 'prompt' in pr.keys():
                        prompt=pr["prompt"]
                        an=pr["answer"]
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                locality_prompts.append(temp_prompts)
                locality_answers.append(temp_answers)
        assert len(prompts)==len(locality_prompts)==len(locality_answers)

        locality_inputs = {}
        portability_inputs = {}
        locality_inputs = {
            'loc_hop':{
                'prompt': locality_prompts,
                'ground_truth': locality_answers
            }
        }
        portability_inputs = {
            'por_hop':{
                'prompt': portability_prompts,
                'ground_truth': portability_answers  
            }       
        }
        return prompts, target_new, ground_truth, subject, rephrase_prompts, locality_inputs, portability_inputs
    prompts, target_new, ground_truth, subject, rephrase_prompts, locality_inputs, portability_inputs = get_data()
    # 处理参数
    editing_hparams = get_editing_hparams()
    hparams = editing_hparams.from_hparams(f'./hparams/{args.editing_method}/Qwen-1_8B-Chat.yaml')
    if args.target_modules is not None:
        if args.target_modules == 'all':
            pass
        elif args.target_modules in ['mlp', 'attn']:
            target_modules = [m for m in hparams.target_modules if args.target_modules in m]
            hparams.target_modules = target_modules
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
    # pre_edit
    # args.pre_file = f"./pre_edit/{hparams.model_name.split('/')[-1]}_{args.data_type}_pre_edit.json"
    # print(args.pre_file)
    # if args.pre_file is not None and os.path.exists(args.pre_file):
    #     pre_edit = json.load(open(args.pre_file,'r', encoding='utf-8'))
    #     # pre_edit = pre_edit[477:480] # debug
    #     # pre_edit = pre_edit[424:424+1] # debug
    #     # pre_edit = pre_edit[:424] # IKE debug
    #     if args.start_idx_end_idx is not None:
    #         start_idx, end_idx = args.start_idx_end_idx.split(',')
    #         pre_edit = pre_edit[int(start_idx):int(end_idx)]
    #     assert len(pre_edit) == len(prompts)
    # else:
    #     pre_edit = None
    if args.editing_method == 'IKE':
        train_ds = CKnowEditDataset(args.train_data_path)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        encode_ike_facts(sentence_model, train_ds, hparams)
    else:
        train_ds = None
    # 保存结果
    if not os.path.exists(args.metrics_save_dir):
        os.makedirs(args.metrics_save_dir)
    
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
    knb_dict = None
    if args.editing_method == 'LoRA':
        save_name = f'{save_name}_r{hparams.rank}_p{hparams.lora_dropout}'
        save_name = f'{save_name}_rs{hparams.use_rslora}_a{hparams.lora_alpha}'
        save_name = f'{save_name}_b_{hparams.bias}_tr{hparams.target_r}_ir{hparams.init_r}'
    elif args.editing_method == 'KNB':
        hparams.p = args.p
        save_name = args.knb_dict_path.split('/')[-1].replace('.json', '')
        save_name += f'_{args.num_steps}'
        save_name = f'{save_name}_p{hparams.p}_rs{hparams.use_rsknb}_a{hparams.knb_alpha}'
        save_name = f'{save_name}_pd{hparams.knb_dropout}_bias_{hparams.bias}_t_loss{hparams.t_loss}'
        with open(args.knb_dict_path, 'r', encoding='utf-8') as f:
            p_data_weight_layer_knb_dict = json.load(f)
        knb_dict_list = p_data_weight_layer_knb_dict[args.p]
    print(f"Hparams:\n{save_name}")
    # 编辑模型
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.generate_edit(
        prompts=prompts,
        target_new=target_new,
        ground_truth=ground_truth,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        subject = subject,
        train_ds=train_ds,
        keep_original_weight=True,
        pre_file=args.pre_file,
        # pre_edit = pre_edit,
        test_generation = True,
        sequential_edit = False,
        is_post_metrics = args.is_post_metrics,
        file_obj = open(os.path.join(args.metrics_save_dir, f'log/{save_name}_log.json'), encoding='utf-8', mode='w'),
        knb_dict_list = knb_dict_list,
        max_new_tokens_times=3,
    )
    
    json.dump(metrics,
            open(os.path.join(args.metrics_save_dir, f'result/{save_name}.json'), encoding='utf-8', mode='w'), 
            indent=4,
            ensure_ascii=False)