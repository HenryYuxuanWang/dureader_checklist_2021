---
# ***dureader checklist 2021***

---
## 百度2021年语言与智能技术竞赛机器阅读理解 
—— 基于transformers的pytorch baseline

---
## 比赛介绍
Link: https://aistudio.baidu.com/aistudio/competition/detail/66?isFromLuge=true<br><br>
中国计算机学会、中国中文信息学会和百度公司已经于2018至2020年间连续联合举办了三届机器阅读理解评测，极大地推动了中文机器阅读理解技术的发展。然而，当前的机器阅读理解数据集大多都只采用单一的指标来评测模型的好坏，缺乏对模型语言理解能力的细粒度、多维度评测，导致模型的具体缺陷很难被发现和改进。为了解决这个问题，我们建立了细粒度的、多维度的评测数据集，从词汇理解、短语理解、语义角色理解、逻辑推理等多个维度检测模型的不足之处，从而推动阅读理解评测进入“精细化“时代。该数据集中的样本均来自于实际的应用场景，难度大，考察点丰富，覆盖了真实应用中诸多难以解决的问题。<br><br>
China Computer Federation(CCF), Chinese Information Processing Society of China(CIPS), and Baidu Inc. have jointly held the shared task of machine reading comprehension (MRC) from 2018 to 2020, which greatly promoted the development of Chinese machine reading comprehension technology. However, most existing MRC datasets focus on the overall performance of an MRC model, lacking systematic and fine-grained evaluation. This leads to a consequence that the vulnerabilities of the current MRC models are not well-studied. Therefore, we hold the shared task of machine reading comprehension in " 2021 Language and Intelligence Challenge ", that focus on challenging the MRC models from multiple aspects, including understanding of vocabulary, phrase, semantic role, reasoning and so on.

---
## 依赖库
transformers==4.5.1<br>
torch==1.8.1+cu111<br>

---
## 示例
train以及evaluate过程都包含在[main.py](main.py)里面
* **train**<br>
```python
from model_pytorch import Model
from training_args import training_args

train_dev_data_dir = "./dataset"

model = Model()

train_examples = model.processor.get_train_examples(train_dev_data_dir)
dev_examples = model.processor.get_dev_examples(train_dev_data_dir)

train_features = model.prepare_training_features(train_examples)
dev_features = model.prepare_validation_features(dev_examples)

dev_data = {"examples": dev_examples, "features": dev_features}

model.train(train_features, dev_data, args=training_args)
```

* **evaluate**<br>
```python
dev_data_dir = "./test1"

model = Model(path='./outputs')

dev_examples = model.processor.get_dev_examples(dev_data_dir)

dev_features = model.prepare_validation_features(dev_examples)

dev_data = {"examples": dev_examples, "features": dev_features}

model.evaluate(dev_data, args=training_args, prefix='test1')
```
