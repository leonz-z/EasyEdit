import os
from typing import Optional, TextIO, Union, List, Tuple, Dict
from time import time
from tqdm import tqdm
import json
import torch
import numpy as np
import random
from ..models.melo.melo import LORA
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from .utils import _chunks, _prepare_requests, summary_metrics
from .batch_editor import BatchEditor
from ..evaluate import compute_edit_quality, compute_icl_edit_quality, compute_sent_metric
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *
from ..evaluate.evaluate_utils import test_generation_quality

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)
def make_logs():

    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
seed_everything(42)

def gpu_mem_report(func):
    # gpu_mem=24
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        torch.cuda.empty_cache()
        
        return res
    
    return wrapper
  
class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self, hparams: HyperParams):
        assert hparams is not None, 'Error: hparams is None.'
        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        make_logs()
        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            # add: use HUGGINGFACE_CACHE golbal variable
            huggingface_cache = os.environ.get('HUGGINGFACE_CACHE')
            if huggingface_cache:
                self.model_name = os.path.join(huggingface_cache, self.model_name)
                print(f"Using Huggingface cache: {self.model_name}")
            # set device
            device_map = 'auto' if hparams.model_parallel else None
            if hasattr(hparams, 'fp16') and hparams.fp16:
                torch_dtype = torch.float16
            elif hasattr(hparams, 'bf16') and hparams.bf16:
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
            LOG.info(f"Using device:{device_map} torch_dtype:{torch_dtype}")
            # load model and tokenizer
            if 't5' in self.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = T5Tokenizer.from_pretrained(self.model_name)
            elif 'gpt-3.5' in self.model_name.lower():
                self.model, self.tok = None, None
            elif 'gpt' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'llama' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'baichuan' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, trust_remote_code=True, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'chatglm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.unk_token_id = 64787
                # self.tok.pad_token_id = self.tok.eos_token_id
            elif 'internlm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'qwen2' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch_dtype if hparams.alg_name not in ['MEND'] else torch.bfloat16, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
            elif 'qwen' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name,fp32=False,trust_remote_code=True, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
            elif 'mistral' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            else:
                raise NotImplementedError

            if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer)) and (hparams.alg_name not in ['ROME', 'MEMIT']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'
            if self.tok is not None and ('mistral' in self.model_name.lower() or 'llama' in self.model_name.lower() or 'qwen' in self.model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
                self.tok.padding_side = 'right'
        else:
            self.model, self.tok = self.model_name

        if hparams.model_parallel:
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams

    def edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs:  Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             sequential_edit=False,
             verbose=True,
             file_obj: Optional[TextIO] = None,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        test_generation = kwargs.pop('test_generation', False)

        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        if hasattr(self.hparams, 'batch_size') and not BatchEditor.is_batchable_method(self.alg_name):  # For Singleton Editing, bs=1
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'

        if ground_truth is not None:
            ground_truth = [ground_truth,] if isinstance(ground_truth, str) else ground_truth
        else:# Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>'] * (len(prompts))

        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = _prepare_requests(prompts, target_new, ground_truth, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)

        return self.edit_requests(requests, sequential_edit, verbose, test_generation=test_generation, file_obj=file_obj, **kwargs)

    def batch_edit(self,
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


        assert BatchEditor.is_batchable_method(self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'
        # 2024-7-13 locality_inputs portability_inputs
        requests = _prepare_requests(prompts, target_new, ground_truth, rephrase_prompts,
                                          locality_inputs, portability_inputs, **kwargs)
        torch.cuda.empty_cache()
        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'
        all_metrics = []
        for record_chunks in _chunks(requests, self.hparams.batch_size):
            start = time()
            if kwargs.get('knb_dict'):
                knb_dict = kwargs['knb_dict']
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    record_chunks,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None,
                    knb_dict=knb_dict
                )
                torch.cuda.empty_cache()
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    record_chunks,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                )
            exec_time = time() - start
            LOG.info(f"Execution editing took {exec_time}")

            start = time()
            chunk_metrics = []
            for i, request in enumerate(record_chunks):

                metrics = {
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation),
                }
                torch.cuda.empty_cache()
                chunk_metrics.append(metrics)

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
            torch.cuda.empty_cache()
            for i, request in enumerate(record_chunks):
                chunk_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation)
                torch.cuda.empty_cache()
                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {chunk_metrics[i]}"
                    )
            
            LOG.info(f"Evaluation took {time() - start}")
            all_metrics.extend(chunk_metrics)
        return all_metrics, edited_model, weights_copy

    def batch_edit_knb(self,
                   prompts: List[str],
                   target_new: List[str],
                   ground_truth: Optional[List[str]] = None,
                   rephrase_prompts: Optional[List[str]] = None,
                   locality_inputs:  Optional[Dict] = None,
                   portability_inputs: Optional[Dict] = None,
                   file_obj: Optional[TextIO] = None,
                   knb_dict_list=None,
                   verbose=True,
                   **kwargs
                   ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        def edit_evaluation(all_metrics, request, edited_model, idx, eval_metric, test_generation, **kwargs):
            all_metrics[idx].update({
                'case_id': idx,
                "requested_rewrite": request,
                "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation),
            })
            if "metric_kwargs" in kwargs:
                all_metrics[idx].update(compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok,metric_kwargs=kwargs["metric_kwargs"][idx], device=self.hparams.device))
            if 'locality' in all_metrics[idx]['post'].keys():
                for locality_key in request['locality'].keys():
                    locality_result = []
                    for ans, label in zip(all_metrics[idx]['post']['locality'][f'{locality_key}_output'], all_metrics[idx]['pre']['locality'][f'{locality_key}_output']):
                        locality_result.append(np.mean(np.equal(ans, label)))
                    all_metrics[idx]['post']['locality'][f'{locality_key}_acc'] = locality_result
                    all_metrics[idx]['post']['locality'].pop(f'{locality_key}_output')
                all_metrics[idx]['pre'].pop('locality')
            file_obj.writelines(json.dumps(all_metrics[idx], ensure_ascii=False) + '\n')
            file_obj.flush()
            if verbose:
                LOG.info(f"{idx} editing: {request['prompt']} -> {request['target_new']}  \n\n {all_metrics[idx]}")
        
        # 处理参数并判断参数是否正确
        assert len(prompts) == len(target_new)
        # assert BatchEditor.is_batchable_method(self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'
        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'
        eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]
        
        # pre_edit
        all_metrics=[]
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        else:
            for i, request in enumerate(tqdm(requests)):
                metrics = {"pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)}
                all_metrics.append(metrics)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)
        requests = _prepare_requests(prompts, target_new, ground_truth, rephrase_prompts,
                                    locality_inputs, portability_inputs, **kwargs)
        
        # apply_knb_to_model
        for i, idx in tqdm(enumerate(range(0, len(requests), self.hparams.batch_size)), total=int(len(requests)/self.hparams.batch_size)):
            end_idx = min(idx+self.hparams.batch_size, len(requests))
            request_batch = requests[idx:end_idx]
            if isinstance(knb_dict_list, list):
                knb_dict = knb_dict_list[i]
            else:
                knb_dict = knb_dict_list
            # batch knb edit
            start = time()
            edited_model, weights_copy = self.apply_algo(
                idx=idx,
                model=self.model,
                tok=self.tok,
                requests=request_batch,
                hparams=self.hparams,
                copy=False,
                return_orig_weights=False,
                keep_original_weight=True,
                knb_dict=knb_dict
            )
            exec_time = time() - start
            LOG.info(f"Execution editing took {exec_time}")
            # evaluation
            for j, request in enumerate(request_batch):
                new_idx = idx+j
                edit_evaluation(all_metrics, request, edited_model, new_idx, eval_metric, test_generation)

            edited_model.unload()
            del self.model.peft_config # 每个batch request对应不同的knb-dict,重新设置peft config
            torch.cuda.empty_cache()

        return all_metrics

    def edit_requests(self,
             requests,
             sequential_edit=False,
             verbose=True,
             test_generation=False,
             file_obj: Optional[TextIO] = None,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'
        knb_dict_list = kwargs.pop('knb_dict_list', None)
        all_metrics = []
        # pre_edit
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        else:
            for i, request in enumerate(tqdm(requests)):
                if self.alg_name == 'IKE':
                    assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                    metrics = {"pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''], request, self.hparams.device, pre_edit=True)}
                else:
                    metrics = {"pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)}
                all_metrics.append(metrics)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)

        def edit_func(request, idx=None, knb_dict=None, **kwargs):
            if self.alg_name == 'IKE':
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            else:
                edited_model, weights_copy = self.apply_algo(
                    idx=idx,
                    model=self.model,
                    tok=self.tok,
                    requests=request,
                    hparams=self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    knb_dict=knb_dict,
                    **kwargs
                )
                icl_examples = None
            return edited_model, weights_copy, icl_examples
        def edit_evaluation(all_metrics, request, edited_model, idx, eval_metric, test_generation, icl_examples, **kwargs):
            if self.alg_name == 'IKE':
                all_metrics[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples, request, self.hparams.device),
                })
            else:
                all_metrics[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation),
                })
                if "metric_kwargs" in kwargs:
                    all_metrics[idx].update(compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok,metric_kwargs=kwargs["metric_kwargs"][idx], device=self.hparams.device))
                if 'locality' in all_metrics[idx]['post'].keys():
                    for locality_key in request['locality'].keys():
                        locality_result = []
                        for ans, label in zip(all_metrics[idx]['post']['locality'][f'{locality_key}_output'], all_metrics[idx]['pre']['locality'][f'{locality_key}_output']):
                            locality_result.append(np.mean(np.equal(ans, label)))
                        all_metrics[idx]['post']['locality'][f'{locality_key}_acc'] = locality_result
                        all_metrics[idx]['post']['locality'].pop(f'{locality_key}_output')
                    all_metrics[idx]['pre'].pop('locality')
            file_obj.writelines(json.dumps(all_metrics[idx], ensure_ascii=False) + '\n')
            file_obj.flush()
            if verbose:
                LOG.info(f"{idx} editing: {request['prompt']} -> {request['target_new']}  \n\n {all_metrics[idx]}")


        if sequential_edit:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
            for i, request in enumerate(requests):
                edit_evaluation(all_metrics, request, edited_model, i, eval_metric, test_generation, icl_examples, **kwargs)
        # single edit
        else:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                knb_dict = knb_dict_list
                edited_model, weights_copy, icl_examples = edit_func(request, i, knb_dict=knb_dict, **kwargs)
                edit_evaluation(all_metrics, request, edited_model, i, eval_metric, test_generation, icl_examples, **kwargs)
                if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                    with torch.no_grad():
                        weights_copy()
                elif self.alg_name in ['LoRA', 'KNB']:
                    edited_model.unload()
                    del self.model.peft_config
                elif self.alg_name == 'MELO':
                    self.model = edited_model
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        if isinstance(edited_model, LORA):
            edited_model = edited_model.model
        # if len(all_metrics) != 0:
        #     summary_metrics(all_metrics)

        return all_metrics, edited_model, weights_copy

    def normal_edit(
        self,
        prompts: List[str],
        target_new: List[str],
        sequential_edit=False,
    ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        # assert BatchEditor.is_batchable_method(self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'

        requests = _prepare_requests(prompts, target_new, ground_truth)

        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'

        # print(f"[editor.py][batch_edit] `batch_size`={self.hparams.batch_size}")
        # for epc in range(epoch):
        #     print(f"[editor.py][batch_edit] `Epoch` = {epc+1}")
        #     for record_chunks in self._chunks(requests, self.hparams.batch_size):
        start = time()

        edited_model, weights_copy = self.apply_algo(
            self.model,
            self.tok,
            requests,  # record_chunks -> requests
            self.hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=False,
        )
        exec_time = time() - start
        LOG.info(f"Execution editing took {exec_time}")

        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        return None, edited_model, weights_copy
    
    def generate_edit(
        self,
        prompts: Union[str, List[str]],
        target_new: Union[str, List[str]],
        ground_truth: Optional[Union[str, List[str]]] = None,
        rephrase_prompts: Optional[Union[str, List[str]]] = None,
        locality_inputs:  Optional[Dict] = None,
        portability_inputs: Optional[Dict] = None,
        sequential_edit: bool=False,
        verbose: bool=True,
        is_post_metrics: bool = False,
        file_obj: Optional[TextIO] = None,
        max_new_tokens_times: int = 3,
        system_prompt: str = "你是一个乐于助人的助手。 ",
        **kwargs
    ):
        eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        test_generation = kwargs.pop('test_generation', False)
        knb_dict_list = kwargs.pop('knb_dict_list', None)
        assert len(prompts) == len(target_new) 
        
        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = _prepare_requests(prompts, target_new, ground_truth, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)
        
        def text_generate(
            model,
            tok: AutoTokenizer,
            query,
            device,
            max_new_tokens = 512,
            system_prompt = "你是一个乐于助人的助手。 "
        ):
            model.eval()
            messages = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            text = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tok.encode(text, return_tensors="pt").to(f"cuda:{device}")
            template_length = len(model_inputs[0])
            generated_ids = model.generate(
                input_ids=model_inputs,
                max_new_tokens=max_new_tokens, 
            )
            trimmed_generated_ids = generated_ids[0][template_length:]
            response = tok.decode(trimmed_generated_ids, skip_special_tokens=True)
            return response

        all_results = []
        # if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
        #     results = kwargs['pre_edit']
        #     all_results = results
        # else:
        #     for i, request in enumerate(tqdm(requests)):
        #         results = {}
        #         results['pre'] = {}
        #         results['pre']['rewrite_ans'] = text_generate(self.model, self.model_name, self.hparams, self.tok, request['prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
        #         results['pre']['rephrase_ans'] = text_generate(self.model, self.model_name, self.hparams, self.tok, request['rephrase_prompt'], self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
        #         por_results = []
        #         for pr in request['portability']['por_hop']['prompt']:
        #             por_results.append(text_generate(self.model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
        #         if 'locality' in request.keys() and 'loc_hop' in request['locality'].keys():
        #             loc_results = []
        #             for pr in request['locality']['loc_hop']['prompt']:
        #                 loc_results.append(text_generate(self.model, self.model_name, self.hparams, self.tok, pr, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation))
        #             results['pre']['locality_ans'] = loc_results
        #         results['pre']['portability_ans'] = por_results
        #         all_results.append(results)
        #     if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
        #         json.dump(all_results, open(kwargs['pre_file'], 'w'), indent=4)

        @gpu_mem_report
        def edit_func(request, idx=None, knb_dict=None, **kwargs):
            if not isinstance(request, list):
                request = [request]
            if self.alg_name == 'IKE':
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            else:
                edited_model, weights_copy = self.apply_algo(
                    idx=idx,
                    model=self.model,
                    tok=self.tok,
                    requests=request,
                    hparams=self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    knb_dict=knb_dict,
                    **kwargs
                )
               
                icl_examples = None
            return edited_model, weights_copy, icl_examples
        
        @gpu_mem_report
        def post_edit_results(all_results, request, edited_model, idx, eval_metric, test_generation, icl_examples, **kwargs):
            if self.alg_name == 'IKE':
                all_results[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples, request, self.hparams.device),
                })
            else:
                # support batch edit post metrics
                if isinstance(request, list):
                    for i in range(len(request)):
                        results_post = {}
                        answer_token_len = self.tok.encode(request[i]['target_new'], return_tensors="pt").shape[1]
                        answer_token_len = min(answer_token_len*max_new_tokens_times, 512)
                        results_post['rewrite_ans'] = text_generate(edited_model, 
                                                                    self.tok, request[i]['prompt'], self.hparams.device,
                                                                     max_new_tokens=answer_token_len, system_prompt=system_prompt)
                        results_post['rephrase_ans'] = text_generate(edited_model, 
                                                                     self.tok, request[i]['rephrase_prompt'], self.hparams.device,
                                                                      max_new_tokens=answer_token_len, system_prompt=system_prompt)
                        por_results = []
                        for pr in request[i]['portability']['por_hop']['prompt']:
                            pr_token_len = self.tok.encode(pr, return_tensors="pt").shape[1]
                            pr_token_len = min(pr_token_len*max_new_tokens_times, 512)
                            por_results.append(text_generate(edited_model, 
                                                             self.tok, pr, self.hparams.device,
                                                             max_new_tokens=pr_token_len, system_prompt=system_prompt))
                        if 'locality' in request[i].keys() and 'loc_hop' in request[i]['locality'].keys():
                            loc_results = []
                            for pr in request[i]['locality']['loc_hop']['prompt']:
                                loc_token_len = self.tok.encode(pr, return_tensors="pt").shape[1]
                                loc_token_len = min(loc_token_len*max_new_tokens_times, 512)
                                loc_results.append(text_generate(edited_model, 
                                                                 self.tok, pr, self.hparams.device,
                                                                 max_new_tokens=loc_token_len, system_prompt=system_prompt))
                            results_post['locality_ans'] = loc_results
                        results_post['portability_ans'] = por_results
                        if test_generation:
                            if self.hparams.alg_name == 'GRACE':
                                results_post['fluency'] = test_generation_quality(model=edited_model,tok=self.tok,prefixes=[request[i]['prompt'],], max_out_len=100, vanilla_generation=True)
                                results_post['fluency_new'] = test_generation_quality(model=edited_model,tok=self.tok,prefixes=[request[i]['prompt'],], max_out_len=answer_token_len, vanilla_generation=True)
                            else:
                                results_post['fluency'] = test_generation_quality(model=edited_model,tok=self.tok,prefixes=[request[i]['prompt'],], max_out_len=100, vanilla_generation=False)
                                results_post['fluency_new'] = test_generation_quality(model=edited_model,tok=self.tok,prefixes=[request[i]['prompt'],], max_out_len=answer_token_len, vanilla_generation=False)
                        # all_results[i+idx].update({
                        #     'case_id': i+idx,
                        #     "requested_rewrite": request[i],
                        #     "post": results_post
                        # })
                        all_results.append({
                            "case_id": i+idx,
                            "requested_rewrite": request[i],
                            "post": results_post
                        })
                        file_obj.writelines(json.dumps(all_results[i+idx], ensure_ascii=False) + '\n')
                        file_obj.flush()
                        if verbose:
                            LOG.info(f"{idx} editing: {request[i]['prompt']} -> {request[i]['target_new']}")
                else:
                    results_post = {}
                    answer_token_len = self.tok.encode(request['target_new'], return_tensors="pt").shape[1]
                    answer_token_len = min(answer_token_len*max_new_tokens_times, 512)
                    results_post['rewrite_ans'] = text_generate(edited_model, 
                                                                self.tok, request['prompt'], self.hparams.device,
                                                                max_new_tokens=answer_token_len, system_prompt=system_prompt)
                    results_post['rephrase_ans'] = text_generate(edited_model, 
                                                                 self.tok, request['rephrase_prompt'], self.hparams.device,
                                                                  max_new_tokens=answer_token_len, system_prompt=system_prompt)
                    por_results = []
                    for pr in request['portability']['por_hop']['prompt']:
                        pr_token_len = self.tok.encode(pr, return_tensors="pt").shape[1]
                        pr_token_len = min(pr_token_len*max_new_tokens_times, 512)
                        por_results.append(text_generate(edited_model, 
                                                         self.tok, pr, self.hparams.device,
                                                        max_new_tokens=pr_token_len, system_prompt=system_prompt))
                    if 'locality' in request.keys() and 'loc_hop' in request['locality'].keys():
                        loc_results = []
                        loc_token_len = self.tok.encode(pr, return_tensors="pt").shape[1]
                        loc_token_len = min(loc_token_len*max_new_tokens_times, 512)
                        for pr in request['locality']['loc_hop']['prompt']:
                            loc_results.append(text_generate(edited_model, 
                                                             self.tok, pr, self.hparams.device,
                                                            max_new_tokens=loc_token_len, system_prompt=system_prompt))
                        results_post['locality_ans'] = loc_results
                    results_post['portability_ans'] = por_results
                    if test_generation:
                        if self.hparams.alg_name == 'GRACE':
                            results_post['fluency'] = test_generation_quality(model=edited_model,tok=self.tok,prefixes=request['prompt'] if isinstance(request['prompt'],list) else [request['prompt'],], max_out_len=100, vanilla_generation=True)
                        else:
                            results_post['fluency'] = test_generation_quality(model=edited_model,tok=self.tok,prefixes=request['prompt'] if isinstance(request['prompt'],list) else [request['prompt'],], max_out_len=100, vanilla_generation=False)
                    # all_results[idx].update({
                    #     'case_id': idx,
                    #     "requested_rewrite": request,
                    #     "post": results_post
                    # })
                    all_results.append({
                        "case_id": idx,
                        "requested_rewrite": request,
                        "post": results_post
                    })
                    file_obj.writelines(json.dumps(all_results[idx], ensure_ascii=False) + '\n')
                    file_obj.flush()
                    if verbose:
                        LOG.info(f"{idx} editing: {request['prompt']} -> {request['target_new']}")

        if sequential_edit:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
            for i, request in enumerate(requests):
                post_edit_results(all_results, request, edited_model, i, eval_metric, test_generation, icl_examples, **kwargs)
        else:
            # DONE:batch edit
            if hasattr(self.hparams, 'batch_size') and self.hparams.batch_size > 1:
                for i, idx in tqdm(enumerate(range(0, len(requests), self.hparams.batch_size)), total=int(len(requests)/self.hparams.batch_size)):
                    knb_dict = knb_dict_list
                    # if knb_dict_list is not None:
                    #     knb_dict = knb_dict_list[i]
                    end_idx = idx+self.hparams.batch_size if idx+self.hparams.batch_size < len(requests) else len(requests)
                    request_batch = requests[idx:end_idx]
                    edited_model, weights_copy, icl_examples = edit_func(request_batch, idx, knb_dict=knb_dict, **kwargs)
                    if is_post_metrics:
                        post_edit_results(all_results, request_batch, edited_model, idx, eval_metric, test_generation, icl_examples, **kwargs)
                    if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                        with torch.no_grad():
                            weights_copy()
                    elif self.alg_name == 'LoRA':
                        edited_model.unload()
                        del self.model.peft_config
                    elif self.alg_name == 'MELO':
                        self.model = edited_model
                    else:
                        with torch.no_grad():
                            for k, v in weights_copy.items():
                                nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

                    # batch edit后暂不评测
            else: # 无batch_size参数或者=1
                for idx, request in enumerate(tqdm(requests, total=len(requests))):
                    knb_dict = knb_dict_list
                    # if knb_dict_list is not None:
                    #     knb_dict = knb_dict_list[idx]
                    edited_model, weights_copy, icl_examples = edit_func(request, idx, knb_dict=knb_dict, **kwargs)
                    post_edit_results(all_results, request, edited_model, idx, eval_metric, test_generation, icl_examples, **kwargs)
                    if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                        with torch.no_grad():
                            weights_copy()
                    elif self.alg_name == 'LoRA':
                        edited_model.unload()
                        del self.model.peft_config
                    elif self.alg_name == 'MELO':
                        self.model = edited_model
                    else:
                        with torch.no_grad():
                            for k, v in weights_copy.items():
                                nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        if isinstance(edited_model, LORA):
            edited_model = edited_model.model
        
        # summary_metrics(all_results)

        return all_results, edited_model, weights_copy


