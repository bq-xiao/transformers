import torch
from torch import nn
from transformers import BertTokenizer, BartForConditionalGeneration, \
    TranslationPipeline, GPT2Config, GPT2Model

gpt = GPT2Config()
model = GPT2Model(gpt)
print(model)

tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese", cache_dir="../download_model/bart-base-chinese")
#print(tokenizer)

model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese", cache_dir="../download_model/bart-base-chinese")
self_attn = model.model.encoder.layers[0].self_attn
print(self_attn)
input = tokenizer("你好", return_tensors="pt")

emb = nn.Embedding(51271, 768)
line = nn.Linear(768, 51271)
with torch.no_grad():
    x = emb(input.input_ids)
    for i in range(10):
        x, attn_weights, _ = self_attn(x)
    hidden_states = line(x)
    out =  hidden_states.argmax(-1)
    out = tokenizer.batch_decode(out)
    print(out)
    logits = model(input.input_ids, attention_mask=input.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentences = tokenizer.batch_decode(predicted_ids)
print(predicted_sentences)
text2text_generator = TranslationPipeline(model, tokenizer)
print(text2text_generator("你好"))