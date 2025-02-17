from copy import deepcopy
import os
from typing import Any, Dict, List, Tuple
from peft import get_peft_model, AdaLoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig, PeftModel
# import sys
# sys.path.append('../../../')
# from .peft import get_peft_model, AdaLoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .lora_hparams import LoRAHyperParams

def gpu_mem_report(func):
    # gpu_mem=24
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        torch.cuda.empty_cache()
        
        return res
    
    return wrapper

@gpu_mem_report
def apply_lora_to_model(
        idx,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: LoRAHyperParams,
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

    edited_model = execute_lora(idx, model, tok, requests, hparams, keep_original_weight, **kwargs)
    
    return edited_model, weights_copy

@gpu_mem_report
def lora_forward(peft_model, txt, tgt, device, tok, loss_meter, opt):
    mask_token = -100
    opt.zero_grad()
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
        idx = None,
        model: AutoModelForCausalLM=None,
        tok: AutoTokenizer=None,
        requests: List[Dict]=None,
        hparams: LoRAHyperParams=None,
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
            peft_config = Config(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hparams.rank,
            lora_alpha=hparams.lora_alpha, lora_dropout=hparams.lora_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules, # target_knb
            knb_dict=knb_dict,
        )
        else:
            # kwargs = {}
            peft_config = Config(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=hparams.rank,
                lora_alpha=hparams.lora_alpha, lora_dropout=hparams.lora_dropout,
                layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
                target_modules=hparams.target_modules,
                use_rslora=hparams.use_rslora,
                use_dora=hparams.use_dora,
                bias=hparams.bias,
                target_r = hparams.target_r,
                init_r = hparams.init_r,
                deltaT = hparams.deltaT,
                beta1 = hparams.beta1,
                beta2 = hparams.beta2,
                orth_reg_weight = hparams.orth_reg_weight,
                # **kwargs, # AdaLoRA的超参数
            )
        peft_model = get_peft_model(model, peft_config)
    assert isinstance(peft_model, PeftModel), f"peft_model{type(peft_model)} is not of type PeftModel"
    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    # 第一次能使用, 第二次就不能用了 应该是第二次调用时,得到的就不是PeftModel了
    # AttributeError: 'QWenLMHeadModel' object has no attribute 'print_trainable_parameters'
    peft_model.print_trainable_parameters()
    requests = deepcopy(requests)
    # for request in requests:
    #     print(
    #         f"Executing LoRA algo for: "
    #         f"[{request['prompt']}] -> [{request['target_new']}]"
    #     )
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
    loss_meter.reset()
    for it in range(hparams.num_steps):
        print(f"Epoch: {it}", end=' ')

        for txt, tgt in zip(
                chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            if 't5' in hparams.model_name.lower():
                mask_token = -100
                opt.zero_grad()
                inputs = tok(txt, return_tensors="pt", padding=True).to(device)
                bs = inputs["input_ids"].shape[0]
                target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                    device
                )
                inputs['labels'] = target_ids
                logits = peft_model(**inputs).logits
                unmasked_log_probs = logits.log_softmax(-1).gather(-1, inputs['labels'].unsqueeze(-1)).squeeze(-1)
                mask = inputs['labels'] != -100
                n_tokens = mask.float().sum()
                avg_log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                nll = -avg_log_prob
                loss = nll
            else:
                lora_forward(peft_model, txt, tgt, device, tok, loss_meter, opt)

                # src_trg_inputs = tok(txt + tgt, return_tensors="pt", padding=True).to(device)
                # bs = src_trg_inputs["input_ids"].shape[0]
                # targ = deepcopy(src_trg_inputs['input_ids'])
                # pred = peft_model(**src_trg_inputs).logits
                # pred = pred[:, :-1]
                # targ = targ[:, 1:]
                # mask = targ != -100
                # n_tokens = mask.float().sum()
                # unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)
                # log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                # loss = -log_prob
                # eos_token = tok.decode(tok.eos_token_id)
                # full_prompt = [f"{p} {l}" for p, l in zip(txt, tgt)]
                # prompt_ids = tok(list(txt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
                # num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_ids]
                # tokens = tok(full_prompt, return_tensors="pt", padding=True, truncation=True)
                # bs = tokens["input_ids"].shape[0]
                # tokens["labels"] = tokens["input_ids"].clone()
                # num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in tokens["labels"]]
                # for i in range(len(txt)):
                #     tokens["labels"][i][num_pad_toks[i]:num_pad_toks[i]+num_prompt_toks[i]] = mask_token
                # tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = mask_token
                # tokens = tokens.to(device)
                # pred = peft_model(**tokens)
                # loss = pred.loss
                
                # pred = peft_model(**tokens)
                # loss = pred.loss
                # targ = target_ids
                # pred = peft_model(**src_trg_inputs).logits
                # pred = pred[:, :-1]
                # pred = pred[:, -targ.size(1):]

                # mask = targ != -100
                # n_tokens = mask.float().sum()
                # unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)
                # log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                # loss = -log_prob
        if hparams.batch_size > 1:
            if (it+1)%20 == 0 or loss_meter.val < 1e-3:
                ckp_path = f'/share/ccks2024_output/checkpoints_{hparams.batch_size}_{hparams.num_steps}/'
                ckp_path += f'{idx}_{idx+hparams.batch_size}_{it+1}_{hparams.alg_name}_CKnowEdit_{hparams.model_name}'
                ckp_path += f'_{hparams.layers[0]}_{hparams.layers[-1]}'
                ckp_path += f'_{"_".join(hparams.target_modules)}'
                ckp_path += f'_r_{hparams.rank}_a_{hparams.lora_alpha}_p_{hparams.lora_dropout}'
                ckp_path += f'_rs_{hparams.use_rslora}_bias_{hparams.bias}'
                ckp_path += f'_tr_{hparams.target_r}_ir_{hparams.init_r}'
                # ckp_path += f'_dt_{hparams.deltaT}'
                # ckp_path += f'_b1_{hparams.beta1}_b2_{hparams.beta2}'
                # ckp_path += f'_or_{hparams.orth_reg_weight}'
                if not os.path.exists(ckp_path):
                    os.makedirs(ckp_path)
                peft_model.save_pretrained(ckp_path)                
        if loss_meter.val < 1e-3:
            print(f"Epoch: {it} Batch loss {loss_meter.val}")
            break

        if (it+1)%10 == 0:
            print(f"Epoch: {it} Total loss {loss_meter.avg}")
            loss_meter.reset()
            
    
    return peft_model


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