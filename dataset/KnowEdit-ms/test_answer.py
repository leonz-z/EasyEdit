import transformers
import torch
import json
import os
# os.environ['CUDA_VISIBLE_DEVICES']= '0'
model_name = 'Meta-Llama-3-8B-Instruct'
# model_name = 'gpt-j-6b'
huggingface_cache = os.environ.get('HUGGINGFACE_CACHE')
model_id =huggingface_cache+model_name

pipeline = transformers.pipeline("text-generation", 
                                 model=model_id, 
                                 model_kwargs={"torch_dtype": torch.bfloat16}, 
                                 device_map="auto",)

data_path = './benchmark_wiki_counterfact_test_cf.json'
# data_path = './benchmark_ZsRE_ZsRE-test-all.json'

with open(data_path, 'r') as f:
    data_list = json.load(f)
    
dataset_type = 'counterfact'
with open(f'answer-{dataset_type}-{model_name}-<=n+1-words.json', 'a') as f:
    for data in data_list[:500]:
        prompt1 = data['prompt']
        answer_len = len(data['ground_truth'].split(' '))+1
        # prompt2 = f"Please response the input in {answer_len} words!\nInput:{data['prompt']}\nResponse:"
        # prompt3 = f"Please answer the question in {answer_len} words!\nQuestion:{data['prompt']}\nAnswer:"
        prompt2 = f"Please response the input in no more than {answer_len} words!\nInput:{data['prompt']}\nResponse:"
        prompt3 = f"Please answer the question in no more than {answer_len} words!\nQuestion:{data['prompt']}\nAnswer:"
        answer1 = pipeline(prompt1, max_new_tokens=answer_len, do_sample=False)
        answer2 = pipeline(prompt2, max_new_tokens=answer_len, do_sample=False)
        answer3 = pipeline(prompt3, max_new_tokens=answer_len, do_sample=False)
        data_dict = {
            'prompt': data['prompt'],
            'ground_truth': data['ground_truth'],
            'answer_len': answer_len,
            'answer1': answer1[0]['generated_text'].replace(prompt1, ''),
            'answer2': answer2[0]['generated_text'].replace(prompt2, ''),
            'answer3': answer3[0]['generated_text'].replace(prompt3, ''),
        }

        f.write(json.dumps(data_dict, ensure_ascii=False) + '\n')
        f.flush()