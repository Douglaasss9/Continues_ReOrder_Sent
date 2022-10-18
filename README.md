## Continues_ReOrder_Sent

### Introduction

* This Project are made for sentence reordering task, which use the same task setting as ReBART
* There're differences from traditional reordering task like ReBART:
  * We use MacBERT as our base model that MacBERT is pretrained under the enhanced-NSP task as BERT, which gives much help for us to gain a better model performance
  * We use the Token-Overlap rate as an augment value for our prediction, since the Token-Overlap rate indicates the relationship between the two sentence.
  * We use Continues Reordering task for our model training, which means we focus on the NSP task instead of predict the order at one time.

### Data Preparation

* We use the same dataset form as it is shown in ReBART(i.e. ROCstory)
* The example of our dataset are shown as follows:

```json
{"orig_sents": ["0", "4", "2", "1", "3"], "shuf_sents": ["我订购了电脑零件。","我打开了门。","门铃响了。","联邦快递的人让我签收零件。","联邦快递的今天到了。"]}
{"orig_sents": ["0", "3", "2", "4", "1"], "shuf_sents": ["艾莉是一名四年级学生,在数学方面有困难。","艾莉那年的数学成绩获得了85分。","她的妈妈每天都会和她一起做数学抽认卡。","艾莉开始使用抽认卡。","很快,艾莉对数学的信心增强了。"]}
```

* We do not provide dataset downloading method, which means you may need to make your dataset by your own.



### Run

* For running, we provide a bash command:

```shell
bash ./train.sh
```

