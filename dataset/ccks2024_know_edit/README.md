# 数据集介绍
## 初赛训练集
初赛使用KnowEdit中的知识修改数据集ZsRE、知识新增数据集Wiki_recent和知识擦除数据集SafeEdit。

|    任务    |   知识注入  | 知识修改 | 知识擦除 |
|:----------:|:-----------:|:--------:|:--------:|
|   数据集   | Wiki_recent |   ZsRE   | SafeEdit |
|    类型    |     事实    |   问答   |   问答   |
| 训练集大小 |     570     |   10000  |   4050   |
| 测试集大小 |     1266    |   1230   |   1350   |


### Wiki_recent
- 训练集文件recent_train.json，每一个sample为{“subject”：头实体，“prompt”：提示，“target_new”：新知识答案，“portability”（少数sample有）：和新知识有关联的知识提示和答案，包括逻辑关联Logical_Generalization、多跳推理Reasoning、头实体别称Subject_Aliasing，“locality”：与新知识无关的知识提示和答案，包括与prompt无关的头实体事实Relation_Specificity和容易遗忘的事实Forgetfulness}。

- 测试集文件recent_test.json， 每一个sample为{“subject”：头实体，“prompt”：提示，“target_new”：新知识答案，“portability”（少数sample有）：和新知识有关联的知识提示和答案，包括逻辑泛化Logical_Generalization、多跳推理Reasoning、头实体别称Subject_Aliasing，“locality”：与新知识无关的知识提示和答案，包括与prompt无关的头实体事实Relation_Specificity和容易遗忘的事实Forgetfulness，“rephrase”：prompt的同义表达}。

    注意：测试集 除 “target_new” 数据外（大模型知识编辑设定允许使用target_new用于训练），严禁使用 “portability” 、 “locality” 、 “rephrase” 数据进行训练，本次初赛供选手热身熟悉任务设定，因此提供所有数据标签，复赛仅提供target_new数据！

### ZsRE
- 训练集为MEND方法提供的原始训练集eric-mitchell/mend: MEND: Fast Model Editing at Scale (github.com)

- 测试集文件zsre_test_all.json，每一个sample为{“subject”：头实体， “target_new”：新知识答案，“prompt”：提示，“ground_truth”：旧知识，“rephrase_prompt”：prompt的同义表达，“portability”：和新知识有关联的知识提示和答案，包括逻辑泛化Logical_Generalization、多跳推理Reasoning、头实体别称Subject_Aliasing，“locality”：与新知识无关的头实体事实Relation_Specificity }

    注意：测试集 除 “target_new” 数据外（大模型知识编辑设定允许使用target_new用于训练），严禁使用 “portability” 、 “locality” 、 “rephrase_prompt” 数据进行训练，本次初赛供选手热身熟悉任务设定，因此提供所有数据标签，复赛仅提供target_new数据！

### SafeEdit
- 训练集文件SafeEdit_train.json,每个sample为{“adversarial prompt” ：越狱输出，“unsafe generation” ：针对用户输入的不安全回复，“safe generation” ：针对用户输入的安全回复，“knowledge constrain” ：无害问答}

- 测试集文件SafeEdit_test.json,每个sample为{“adversarial prompt” ：越狱输出，“unsafe generation” ：针对用户输入的不安全回复，“safe generation” ：针对用户输入的安全回复，“knowledge constrain” ：无害问答，“generalization test”：与擦除的不安全知识相关的越狱输入}

    注意：SafeEdit数据仅供学术研究使用，不得私自公开，烦请遵守比赛协议！严禁使用 “generalization test”中的数据以及 SafeEdit以外的监督信号进行训练 ， 本次初赛供选手热身熟悉任务设定，因此提供所有数据标签，复赛仅提供target_new数据！

详情见:[CCKS2024——大模型知识编辑评测](https://tianchi.aliyun.com/competition/entrance/532182/information)