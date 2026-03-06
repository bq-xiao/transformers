import json
import re

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from modelscope.models import Model

from transformers import BertTokenizer, GPT2LMHeadModel, GPT2Config
from transformers import Seq2SeqTrainingArguments, \
    Trainer, EarlyStoppingCallback, DefaultDataCollator

tokenizer = BertTokenizer.from_pretrained(r"../download_model/gpt2-distil-chinese-cluecorpussmall")
model_config = GPT2Config(
        vocab_size=len(tokenizer),
        bos_token_id=1,
        eos_token_id=2,
    )
#model = GPT2LMHeadModel.from_pretrained(r"../download_model/gpt2-distil-chinese-cluecorpussmall", config = model_config)
#model = GPT2LMHeadModel.from_pretrained(r"../download_model/gpt2-distil-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained(r"../download_model/gpt2-distil-chinese-cluecorpussmall")
print(model)
# 2. 添加对话特殊标记
special_tokens = {
    'additional_special_tokens': [
        '<user>',
        '<assistant>',
        '<system>',
        '<end>'
    ]
}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# 设置填充标记
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            raw_data = json.loads(line)
            question, answer = raw_data['question'], raw_data['answer']
            str = f"<user>{question}<assistant>{answer}<end>"
            data.append(str)
    return data

file = r'train.jsonl'
conversations = read_jsonl(file)[:100]
dataset = Dataset.from_dict({"text": conversations})

# 4. 数据预处理
def preprocess_function(examples):
    # 对话文本
    texts = examples["text"]

    # 分词处理
    tokenized = tokenizer(
        texts,
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

batch_size = 8
# map
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["text"], batch_size=batch_size)
tokenized_datasets.set_format("torch")
tokenized_datasets = tokenized_datasets.train_test_split(train_size=0.9)

training_args = Seq2SeqTrainingArguments(
    output_dir="chat-gpt2",
    overwrite_output_dir=True,
    num_train_epochs=3, # 对话任务通常需要更多轮次
    per_device_train_batch_size=batch_size,  # 对话序列较长，减小batch size
    per_device_eval_batch_size=batch_size,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    save_strategy="steps",
    eval_strategy="steps",
    load_best_model_at_end=True,
    fp16=True,  # 启用混合精度
    logging_steps=100,
    logging_dir="chat-gpt2/logs",
    report_to=None,
    prediction_loss_only=True,
    save_steps=2000,
    eval_steps=2000,
    save_total_limit=1,
    push_to_hub=False,
)

# lora_config = LoraConfig(
# 	task_type=TaskType.CAUSAL_LM,
# 	r=8,
# 	lora_alpha=32,
# 	lora_dropout=0.1,
#     fan_in_fan_out=True,
# 	target_modules=["attn.c_proj", "attn.c_attn"]
# )
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()  # 打印可训练参数

data_collator = DefaultDataCollator()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    processing_class=tokenizer,
    #data_collator=data_collator,
    #compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

print("start trainning ...")
trainer.train()
model.save_pretrained("GPT2/fine-tuned-model-final")
tokenizer.save_pretrained("GPT2/fine-tuned-model-final")
print("Final model saved successfully")

