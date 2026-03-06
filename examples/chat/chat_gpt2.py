from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    BertTokenizerFast,
)

from trl import DataCollatorForCompletionOnlyLM
from transformers import Trainer, TrainingArguments
from datasets import load_from_disk

import torch

import os

from config import train_args

def main():
    tokenizer = BertTokenizerFast.from_pretrained("gpt2-chatbot-chinese")

    tokenized_datasets = load_from_disk(train_args.dataset_name)
    # 修改默认的属性
    model_config = GPT2Config(
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # 从零开始训练
    model = GPT2LMHeadModel.from_pretrained("gpt2-chatbot-chinese", config=model_config)
    #model.resize_token_embeddings(len(tokenizer))
    #model = GPT2LMHeadModel(model_config)

    device = torch.device(
        f"cuda:{train_args.device}" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    # 定义训练参数
    args = TrainingArguments(
        output_dir="gpt2-chatbot-chinese-output",
        per_device_eval_batch_size=train_args.batch_size,
        per_device_train_batch_size=train_args.batch_size,
        evaluation_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        report_to="none",
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        num_train_epochs=train_args.epochs,
        weight_decay=train_args.weight_decay,
        warmup_steps=train_args.warmup_steps,
        max_grad_norm=train_args.max_grad_norm,
        lr_scheduler_type="linear",
        learning_rate=train_args.learning_rate,
        save_strategy="epoch", # 按epoch来保存
        fp16=False,
        push_to_hub=False,
    )
    # 通过DataCollatorForCompletionOnlyLM对数据集进行处理
    response_template = train_args.bot_token
    instruction_template = train_args.user_token
    data_collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        ignore_index=train_args.ignore_index, # -100
    )

    out = data_collator([tokenized_datasets["train"][i] for i in range(2)])
    for key in out:
        print(f"{key} shape : {out[key].shape}")

    print(tokenizer.batch_decode(out["input_ids"]))

    print(out)
    # 定义训练器
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )
    # 开始训练
    trainer.train()
    # 训练结束，保存最终模型
    trainer.save_model("gpt2-chatbot-chinese-output-finished")


if __name__ == "__main__":
    if not train_args.use_wandb:
        os.environ["WANDB_DISABLED"] = "false"

    main()