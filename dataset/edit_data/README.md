#### Datasets for Factual Knowledge
| **dataset** | Google Drive| BaiduNetDisk | Description |
| :--------: | :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| _ZsRE_ plus | [[Google Drive]](https://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view?usp=sharing) | [[BaiduNetDisk]](https://pan.baidu.com/s/1cQleUMsNjuDk4BKx2bZkag?pwd=xzky) | Question Answering dataset using question rephrasings |
| _Counterfact_ plus | [[Google Drive]](https://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view?usp=sharing) | [[BaiduNetDisk]](https://pan.baidu.com/s/1cQleUMsNjuDk4BKx2bZkag?pwd=xzky) | Counterfact dataset using Entity replacement |


We provide zsre and counterfact datasets to verify the effectiveness of knowledge editing. You can download them here. [[Google Drive]](https://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view?usp=sharing), [[BaiduNetDisk]](https://pan.baidu.com/s/1cQleUMsNjuDk4BKx2bZkag?pwd=xzky).

- For **locality**, in addition to testing unrelated instances, we also provide tests on distracting ([reference: Detecting Edit Failures...](https://arxiv.org/abs/2305.17553)), other attribution, and other downstream tasks (such as commonsense reasoning).
- For **portability**, it tests whether the model can apply edited instances for inference. We provide evaluations for one-hop reasoning, subject alias, and inverse relation (eg, a one-to-one relationship between spouses should be bidirectionally edited).

<details><summary> <b> dataset description </b> </summary>

```text
editing-data
├── counterfact
│   ├── counterfact-edit.json
│   ├── counterfact-train.json
│   └── counterfact-val.json
├── locality
│   ├── Commonsense Task
│   │   ├── piqa_valid-labels.lst
│   │   └── piqa_valid.jsonl
│   ├── Distracting Neighbor
│   │   └── counterfact_distracting_neighbor.json
│   └── Other Attribution
│       └── counterfact_other_attribution.json
├── portability
│   ├── Inverse Relation
│   │   └── zsre_inverse_relation.json
│   ├── One Hop
│   │   ├── counterfact_portability_gpt4.json
│   │   └── zsre_mend_eval_portability_gpt4.json
│   └── Subject Replace
│       ├── counterfact_subject_replace.json
│       └── zsre_subject_replace.json
└── zsre
    ├── zsre_mend_eval.json
    ├── zsre_mend_train_10000.json
    └── zsre_mend_train.json
```

- counterfact: original counterfact dataset using Entity replacement
- zsre: original question answering dataset using question rephrasings
- locality (evaluation for locality, see details in this [paper](https://arxiv.org/abs/2305.13172))
    - Commonsense Task: evaluation for other downstream tasks such as commonsense task
    - Distracting Neighbor: test on distracting neighborhood ([reference: Detecting Edit Failures...](https://arxiv.org/abs/2305.17553))
    - Other Attribution
- portability
    - Inverse Relation: evaluation for one-to-one relationship such as `spouse`
    - One Hop: evaluation for one-hop reasoning
    - Subject Replace: evaluation for synonym replacement
    </details>