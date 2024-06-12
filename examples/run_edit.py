import sys
sys.path.append('..')
import argparse
from easyeditor import ZsreDataset
from sentence_transformers import SentenceTransformer
from easyeditor.models.ike import encode_ike_facts
from easyeditor import BaseEditor
from easyeditor import (
    FTHyperParams,
    IKEHyperParams,
    KNHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams,
    WISEHyperParams,
)
import random
import json
import os.path

def load_counterfact(path):
    """
        {
            "case_id": 2,
            "prompt": "Toko Yasuda, the",
            "target_new": "piano",
            "subject": "Toko Yasuda",
            "ground_truth": "guitar",
            "rephrase_prompt": "Toko Yasuda is incredible at",
            "locality_prompt": "John Lennon performs on the",
            "locality_ground_truth": "guitar"
        },
    """
    test_data = json.load(open(path, 'r', encoding='utf-8'))
    prompts = [test_data_['prompt'] for test_data_ in test_data]
    rephrase_prompts = [edit_data_['rephrase_prompt']
                        for edit_data_ in test_data]
    target_new = [edit_data_['target_new'] for edit_data_ in test_data]
    locality_prompts = [edit_data_['locality_prompt']
                        for edit_data_ in test_data]
    locality_ans = [edit_data_['locality_ground_truth']
                    for edit_data_ in test_data]
    locality_inputs = {
        'neighborhood': {
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }

    subject = [edit_data_['subject'] for edit_data_ in test_data]
    return {
        "prompts":prompts,
        "rephrase_prompts":rephrase_prompts,
        "target_new":target_new,
        "subject":subject,
        "locality_inputs":locality_inputs,
        "portability_inputs":None,
    }

def load_zsre(path):
    """
    {
        "subject": "Christiane Cohendy",
        "src": "What is the native language of Christiane Cohendy?",
        "pred": "French",
        "rephrase": "What's Christiane Cohendy's mother tongue?",
        "alt": "German",
        "answers": [
            "French"
        ],
        "loc": "nq question: what is the most current season of the walking dead",
        "loc_ans": "The eighth season",
        "cond": "French >> German || What is the native language of Christiane Cohendy?"
    }
    """
    test_data = json.load(open(path, 'r', encoding='utf-8'))
    prompts = [test_data_['src'] for test_data_ in test_data]
    rephrase_prompts = [edit_data_['rephrase']
                        for edit_data_ in test_data]
    target_new = [edit_data_['alt'] for edit_data_ in test_data]
    locality_prompts = [edit_data_['loc']
                        for edit_data_ in test_data]
    locality_ans = [edit_data_['loc_ans']
                    for edit_data_ in test_data]
    locality_inputs = {
        'neighborhood': {
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }

    subject = [edit_data_['subject'] for edit_data_ in test_data]
    return {
        "prompts": prompts,
        "rephrase_prompts": rephrase_prompts,
        "target_new": target_new,
        "subject": subject,
        "locality_inputs": locality_inputs,
        "portability_inputs": None,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)

    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'SERACH':
        editing_hparams = SERACHparams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    else:
        raise NotImplementedError

    hparams = editing_hparams.from_hparams(args.hparams_dir)

    if args.editing_method == 'IKE':
        train_data_path = os.path.join(
            args.data_dir, 'zsre_mend_train_10000.json')
        train_ds = ZsreDataset(train_data_path)
        sentence_model = SentenceTransformer(
            hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        encode_ike_facts(sentence_model, train_ds, hparams)
    else:
        train_ds = None

    if "zsre" in args.data_dir.lower():
        dataset = load_zsre(args.data_dir)
    elif "counterfact" in args.data_dir.lower():
        dataset = load_counterfact(args.data_dir)

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        **dataset,
        keep_original_weight=False # 会直接在模型权重文件上修改吗？
    )

    json.dump(metrics, open(args.metrics_save_dir, 'w'), indent=4)