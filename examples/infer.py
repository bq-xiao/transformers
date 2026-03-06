import torch

from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline, GPT2LMHeadModel

from peft import PeftModel

def generate_bart():
    tokenizer = BertTokenizer.from_pretrained(r"../bart_qa/bart_qa/checkpoint-36000")
    model = BartForConditionalGeneration.from_pretrained(r"../bart_qa/bart_qa/checkpoint-36000")
    text2text_generator = Text2TextGenerationPipeline(model, tokenizer)

    question = "如何评价鹿晗在娱乐圈的社交网络？"
    text = text2text_generator(question)
    print(text)


def generate_dialogue(prompt, max_length=512, temperature=0.7, top_p=0.9):
    tokenizer = BertTokenizer.from_pretrained(r"../chat-gpt2/checkpoint-38000")
    model = GPT2LMHeadModel.from_pretrained(r"../download_model/gpt2-distil-chinese-cluecorpussmall")
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

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],  # GPT-2注意力模块
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    #print(model.active_adapters)
    #model.print_trainable_parameters()
    model.load_adapter(r"../chat-gpt2/checkpoint-38000", "default")
    model = model.merge_and_unload()
    #model = PeftModel.from_pretrained(model, r"../chat-gpt2/checkpoint-38000")
    # 格式化输入
    input_text = f"<user> {prompt}<assistant>"

    # 编码输入
    inputs = tokenizer(input_text, return_tensors="pt")

    #attention_mask = torch.ones(inputs.input_ids.shape, dtype=torch.long, device=inputs.input_ids.device)
    # 生成回复
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=21131,
        eos_token_id=tokenizer.convert_tokens_to_ids("<end>"),
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1
    )

    # 解码并提取助理回复
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_start = full_response.find("<assistant>") + len("<assistant>")
    end_marker = full_response.find("<end>", assistant_start)
    if end_marker == -1:
        end_marker = len(full_response)

    assistant_response = full_response[assistant_start:end_marker].strip()
    return assistant_response


# 10. 测试对话
test_prompts = [
    "宋代的都城是哪个城市？",
    "郑和下西洋是哪个朝代的事情？",
    "清朝最后一个皇帝是谁？",
    "第一个女皇帝叫什么？"
]
generate_bart()
# for prompt in test_prompts:
#     response = generate_dialogue(prompt)
#     print(f"用户: {prompt}")
#     print(f"助理: {response}")
#     print("-" * 50)