from transformers import BertTokenizer, BertModel, pipeline, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-chinese", cache_dir="../download_model/bert-base-chinese")
question = "What is the capital of France?"
context = "France is a country in Europe. Its capital is Paris."
tokens = tokenizer.encode_plus(question, context)
print(tokens)
model = BertForMaskedLM.from_pretrained("google-bert/bert-base-chinese", cache_dir="../download_model/bert-base-chinese")
#print(model)
text = "巴黎是[MASK]国的首都。"
fill_masker = pipeline(task="fill-mask", model=model, tokenizer=tokenizer)
output=fill_masker(text)
print(output)
#encoded_input = tokenizer(text, return_tensors='pt')
#output = model(**encoded_input)
#print(output)
#print(output.logits.shape)