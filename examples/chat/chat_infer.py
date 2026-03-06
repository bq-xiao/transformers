# 上面保存的分词器
import re

from config import train_args
from transformers import AutoTokenizer, GPT2LMHeadModel

pattern = r'(?<=<EOS>).+?(?=<EOS>)'

tokenizer = AutoTokenizer.from_pretrained(r"checkpoint")
model = GPT2LMHeadModel.from_pretrained(r"checkpoint")


def generate_dialogue(prompt, max_length=1024, temperature=0.7, top_p=0.9):
    # 格式化输入
    # input_ids = [train_args.bos_token_id]
    # input_ids += (
    #         [train_args.user_token_id]
    #         + tokenizer.encode(prompt, add_special_tokens=False)
    #         + [train_args.eos_token_id]
    # )
    # input = torch.tensor(input_ids, dtype=torch.int64)
    # print(input_ids)
    # print(tokenizer.decode(input, skip_special_tokens=False))
    input_text = f"<BOS><User>{prompt}<EOS>"

    #编码输入
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
    #print(inputs.input_ids.detach().numpy())
    #print(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False))
    # 生成回复
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=train_args.eos_token_id,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1
    )

    # 解码并提取助理回复
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    resp = re.findall(pattern, full_response)
    if len(resp) == 0:
        return ">>>>" + full_response
    assistant_response = resp[0].strip()
    return assistant_response


# 10. 测试对话
test_prompts = [
    "你好吗？",
    "今天天气怎么样？",
    "给我讲个笑话吧",
    "你对人工智能有什么看法？"
]

for prompt in test_prompts:
    response = generate_dialogue(prompt)
    print(f"用户: {prompt}")
    print(f"助理: {response}")
    print("-" * 50)