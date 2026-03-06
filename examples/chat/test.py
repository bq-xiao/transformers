from modelscope.models import Model
from trl import DataCollatorForCompletionOnlyLM

model = Model.from_pretrained(r"nlp_gpt3_text-generation_chinese-base")
print(model)
tokenizer = model.tokenizer

list = ["你最喜欢的颜色是什么？", "我最喜欢的颜色是蓝色。", "你喜欢户外活动吗？", "是的，我喜欢徒步和骑行。"]
strs = ["<Human>你最喜欢的颜色是什么？<Assistant>我最喜欢的颜色是蓝色。", "<Human>你喜欢户外活动吗？<Assistant>是的，我喜欢徒步和骑行。"]

out = tokenizer.batch_encode_plus(strs)
print(out)
dec = tokenizer.batch_decode(out['input_ids'])
print(dec)

# print(len(tokenizer))
special_tokens = {
    'additional_special_tokens': [
        '<Human>',
        '<Assistant>'
    ]
}
i = tokenizer.add_special_tokens(special_tokens)
print(i)
out = tokenizer.batch_encode_plus(strs, max_length=128, padding="max_length", return_tensors='pt', truncation=True)
print(out)
dec = tokenizer.batch_decode(out['input_ids'])
print(dec)
print("=" * 100)
# print(len(tokenizer))
str = "<Human>你最喜欢的颜色是什么？<Assistant>我最喜欢的颜色是蓝色。"
enc = tokenizer.encode(str)
print(enc)
print(tokenizer.decode(enc))
print("=" * 100)
response_template = "<Assistant>"
instruction_template = "<Human>"
data_collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        #ignore_index=train_args.ignore_index, # -100
    )
out = data_collator([enc])
print(out)
print(tokenizer.batch_decode(out['input_ids']))
print(tokenizer.batch_decode(out['labels']))