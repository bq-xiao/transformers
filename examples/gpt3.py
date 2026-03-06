import json
import re

from datasets import Dataset
from modelscope import Preprocessor, MsDataset, Model


def read_jsonl(file_path):
    src_txt, tgt_txt = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            raw_data = json.loads(line)
            question, answer = raw_data['question'], raw_data['answer']
            src_txt.append(question)
            tgt_txt.append(answer)
    return src_txt, tgt_txt

file = r'train.jsonl'
src_txt, tgt_txt = read_jsonl(file)
dataset = Dataset.from_dict({"src_txt": src_txt, "tgt_txt":tgt_txt})
# 模拟训练数据集
src_dataset_dict = {
    'src_txt': [
        '测试文本1', '测试文本2', '测试文本3'
    ]
}
tokenizer = Preprocessor.from_pretrained(r"../download_model/nlp_gpt3_text-generation_chinese-base")
model = Model.from_pretrained(r"../download_model/nlp_gpt3_text-generation_chinese-base")
# 2. 添加对话特殊标记
special_tokens = {
    'additional_special_tokens': [
        '<user>',
        '<assistant>',
        '<system>',
        '<end>'
    ]
}
# tokenizer.add_special_tokens(special_tokens)
# model.resize_token_embeddings(len(tokenizer))
#
# # 设置填充标记
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    # 对话文本
    texts = examples["src_txt"]

    # 分词处理
    tokenized = tokenizer(
        examples,
        truncation=True,
        max_length=512,  # 对话通常较长
        padding="max_length",
        return_tensors="pt"
    )

    # 创建标签 - 忽略用户输入部分的损失
    input_ids = tokenized["input_ids"]
    labels = input_ids.clone()

    # 标记需要忽略的位置（用户发言部分）
    for i, text in enumerate(texts):
        # 找到所有用户发言的位置
        user_turns = [m.start() for m in re.finditer('<user>', text)]

        # 将用户发言部分的标签设为-100（忽略损失）
        for start in user_turns:
            # 找到用户发言的token位置
            start_text = text[:start]
            start_token = tokenizer(
                start_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,  # 对话通常较长
                padding="max_length",
                add_special_tokens=False
            ).input_ids.shape[1]

            # 找到发言结束位置（下一个特殊标记）
            end = text.find('<', start + 1)
            if end == -1:
                end = len(text)

            end_text = text[:end]
            end_token = tokenizer(
                end_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,  # 对话通常较长
                padding="max_length",
                add_special_tokens=False
            ).input_ids.shape[1]

            # 将用户发言部分的标签设为-100
            labels[i, start_token:end_token] = -100
            #print(labels)

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }
tokenized_datasets = dataset.map(preprocess_function, batched=True, batch_size=1, num_proc=1)
tokenized_datasets.set_format("torch")
tokenized_datasets = tokenized_datasets.train_test_split(train_size=0.9)
train_dataset = MsDataset(tokenized_datasets['train'])
eval_dataset = MsDataset(tokenized_datasets['test'])

# 基于modelscope中文gpt3底座二次开发得到诗词生成模型代码
from modelscope.trainers import build_trainer
from modelscope.metainfo import Trainers

num_warmup_steps = 100
def noam_lambda(current_step: int):
    current_step += 1
    return min(current_step ** (-0.5), current_step * num_warmup_steps ** (-1.5))

def cfg_modify_fn(cfg):
    cfg.train.lr_scheduler = {
        "type": "LambdaLR",
        "lr_lambda": noam_lambda,
        "options": {"by_epoch": False}
    }
    cfg.train.optimizer = {
        "type": "AdamW",
        "lr": 3e-4
    }
    cfg.train.dataloader = {"batch_size_per_gpu": 16, "workers_per_gpu": 1}
    return cfg

kwargs = dict(
    model='iic/nlp_gpt3_text-generation_chinese-base',
    train_dataset=train_dataset,
    eval_datase=eval_dataset,
    max_epochs=3,
    work_dir="gpt3",
    cfg_modify_fn=cfg_modify_fn)

# 构造 trainer 并进行训练
trainer = build_trainer(
    name=Trainers.nlp_base_trainer, default_args=kwargs)
trainer.train()
