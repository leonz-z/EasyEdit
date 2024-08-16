from copy import deepcopy
import os
import random
from typing import Any, Dict, List, Tuple
from .peft import get_peft_model, TaskType, KnbConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .knb_hparams import KNBHyperParams

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
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        torch.cuda.empty_cache()
        return res
    return wrapper

@gpu_mem_report
def apply_knb_to_model(
        idx,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: KNBHyperParams,
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

    edited_model = execute_knb(idx, model, tok, requests, hparams, keep_original_weight, **kwargs)

    return edited_model, weights_copy

@gpu_mem_report
def knb_forward(peft_model, txt, tgt, device, tok, loss_meter, opt):
    mask_token = -100
    opt.zero_grad()
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
        idx,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: KNBHyperParams,
        keep_original_weight=False,
        **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the KNB update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    model.config.use_cache = False
    model.supports_gradient_checkpointing = True
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if not keep_original_weight and hasattr(model,'peft_config'):
        peft_model = model
    else:
        if kwargs.get('knb_dict'):
            knb_dict = kwargs['knb_dict']
        else:
            knb_dict = None
            raise "Error: execute_knb -> No knb_dict provided"
        peft_config = KnbConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            knb_alpha=hparams.knb_alpha, knb_dropout=hparams.knb_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules, # target_knb
            knb_dict=knb_dict,
            use_rsknb=hparams.use_rsknb,
            bias=hparams.bias,
        )
        peft_model = get_peft_model(model, peft_config)

    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    peft_model.print_trainable_parameters()
    requests = deepcopy(requests) # 训练log中观察到每次request都是一个batch_size大小
    for request in requests:
        print(
            f"Executing KNB algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    device = torch.device(f'cuda:{hparams.device}')
    print(f"Using device: {device}")
    # Define inputs
    # texts = [r["prompt"] for r in requests]
    # targets = [r["target_new"] for r in requests]
    texts_targets = [[r["prompt"], r["target_new"]] for r in requests]

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
    t_loss = hparams.t_loss
    for it in range(hparams.num_steps):
        print(f"Epoch: {it}", end=' ')

        # for txt, tgt in zip( # 输入数据为batch_size条,只循环一次
        #         chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        # ):
        # 把texts_targets数据打乱
        random.shuffle(texts_targets)
        for data in chunks(texts_targets, hparams.batch_size):
            txt, tgt = zip(*data)
            knb_forward(peft_model, txt, tgt, device, tok, loss_meter, opt)
            
        # if hparams.batch_size > 1:
        #     if (it+1)%20 == 0 or loss_meter.val < t_loss:
        #         ckp_path = f'/share/ccks2024_output/knb/checkpoints_{hparams.batch_size}_{hparams.num_steps}/'
        #         ckp_path += f'{idx}_{idx+hparams.batch_size}_{it+1}_{hparams.alg_name}_CKnowEdit_{hparams.model_name}'
        #         ckp_path += f'_{hparams.layers[0]}_{hparams.layers[-1]}'
        #         ckp_path += f'_{"_".join(hparams.target_modules)}'
        #         ckp_path += f'_a_{hparams.knb_alpha}_p_{hparams.knb_dropout}'
        #         ckp_path += f'_rs_{hparams.use_rsknb}'

        #         if not os.path.exists(ckp_path):
        #             os.makedirs(ckp_path)
        #         peft_model.save_pretrained(ckp_path)                
        if loss_meter.val < t_loss:
            print(f"Epoch: {it} Batch loss {loss_meter.val} < {t_loss}")
            break

        if (it+1)%10 == 0:
            print(f"Epoch: {it} Total loss {loss_meter.avg}")
            loss_meter.reset()
            
    return peft_model