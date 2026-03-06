from dataclasses import dataclass

@dataclass
class TrainArguments:
    batch_size: int = 16
    weight_decay: float = 1e-1
    epochs: int = 10
    warmup_steps: int = 4000
    learning_rate: float = 5e-5 # 似乎还可以大一点
    logging_steps = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 2.0
    use_wandb: bool = False
    from_remote: bool = False
    dataset_name: str = "chichat_dataset"
    valid_size: float = 0.05
    model_name: str = "gpt2-chatbot-chinese"
    tokenizer_name: str = "gpt2-chatbot-chinese"
    device: str = 0
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    bot_token: str = "<Assistant>"
    user_token: str = "<User>"
    bos_token_id: int = 21128
    eos_token_id: int = 21129
    bot_token_id: int = 21130
    user_token_id: int = 21131
    ignore_index: int = -100

train_args = TrainArguments()

from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_from_disk

def main():
    tokenizer = AutoTokenizer.from_pretrained(r"gpt2-chatbot-chinese-output/checkpoint-12000")
    model = GPT2LMHeadModel.from_pretrained(r"gpt2-chatbot-chinese-output/checkpoint-12000")
    print(model)

    tokenized_datasets = load_from_disk('gpt2-chat-preprocess')
    print(tokenized_datasets)

    from transformers import (
        EarlyStoppingCallback,
    )

    from trl import DataCollatorForCompletionOnlyLM
    from transformers import Trainer, TrainingArguments

    # 定义训练参数
    args = TrainingArguments(
        output_dir="gpt2-chatbot-chinese-output",
        overwrite_output_dir=True,
        per_device_eval_batch_size=train_args.batch_size,
        per_device_train_batch_size=train_args.batch_size,
        eval_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        logging_steps=100,
        logging_dir="gpt2-chatbot-chinese-output/logs",
        report_to="tensorboard",
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        num_train_epochs=train_args.epochs,
        weight_decay=train_args.weight_decay,
        warmup_steps=train_args.warmup_steps,
        max_grad_norm=train_args.max_grad_norm,
        lr_scheduler_type="linear",
        learning_rate=train_args.learning_rate,
        load_best_model_at_end=True,
        save_strategy="steps", # 按epoch来保存
        #fp16=True,
        save_total_limit=1,
        push_to_hub=False,
        dataloader_num_workers = 4,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 开始训练
    print("start trainning ...")
    trainer.train(True)

    # 训练结束，保存最终模型
    print("start save model ...")
    trainer.save_model("gpt2-chatbot-chinese-output-finished")

if __name__ == "__main__":
    main()