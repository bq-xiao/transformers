from typing import Dict

import torch
from modelscope.models.nlp.gpt3 import GPT3Model
from modelscope.models.base import Tensor
from torch import nn
from trl import DataCollatorForCompletionOnlyLM

from transformers import BertTokenizer, PreTrainedModel, PretrainedConfig, EarlyStoppingCallback


class GPT3LMHeadModel(PreTrainedModel):
    #base_model_prefix = "model"
    def __init__(self, model, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.model = model

    def get_input_embeddings(self) -> nn.Module:
        return self.model.language_model.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.model.language_model.word_embeddings = new_embeddings

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.model(**input)

model = GPT3Model.from_pretrained(r"nlp_gpt3_text-generation_chinese-base")
obj_dict = vars(model.config)
config = PretrainedConfig(**obj_dict)
model = GPT3LMHeadModel(model, config)
tokenizer = BertTokenizer.from_pretrained(r"nlp_gpt3_text-generation_chinese-base")
print(model)
special_tokens = {
    'additional_special_tokens': [
        '<Human>',
        '<Assistant>'
    ]
}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
print(model)
# aa = "你好吗？"
# enc = tokenizer.batch_encode_plus([aa], return_tensors="pt")
# # add a dim as [batch_size, seq_len]
# input_ids = enc.input_ids
# attention_mask = enc.attention_mask
# generated = []
# for _ in range(64):
#     input = {
#         'input_ids':input_ids,
#         'attention_mask':attention_mask
#     }
#     outputs = model(input=input)
#     logits = outputs.logits
#     next_token_logits = logits[0, -1, :]
#     next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
#     generated.append(next_token_id.item())
#     if next_token_id.item() == tokenizer.eos_token_id:
#         break
#     b =  next_token_id.unsqueeze(0)
#     input_ids = torch.cat((input_ids, b), dim=1)
#     attention_mask = torch.cat((attention_mask, torch.tensor([1]).unsqueeze(0)), dim=1)
#
# output_text = tokenizer.decode(
#     generated,
#     skip_special_tokens=True,
#     clean_up_tokenization_spaces=True,
# )
# print(output_text)
print("--" * 100)
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
# text_generation_zh = pipeline(Tasks.text_generation, model=r'nlp_gpt3_text-generation_chinese-base', external_engine_for_llm=False)
# result_zh = text_generation_zh("随着计算机视觉的飞速发展,人脸识别技术已从简单场景发展到复杂场景,也即姿态、光照、表情、噪声、遮挡、化妆、年龄、种族、性别等差异化所呈现的复杂场景。尽管已有的人脸识别系统在特定约束环境下的识别成功率较高,")
# print(result_zh['text'])

from transformers import Trainer, TrainingArguments

# 定义训练参数
training_args = TrainingArguments(
    output_dir="bart_translation",
    do_eval=True,
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end = True,
    #auto_find_batch_size = True,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    push_to_hub=False,
    report_to='tensorboard',
    logging_dir="logs",
    logging_strategy="steps",
    logging_steps=100,
)
response_template = "<Assistant>"
instruction_template = "<Human>"
data_collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        #ignore_index=train_args.ignore_index, # -100
    )

# 定义训练器
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# 开始训练
print("start trainning ...")
trainer.train()


