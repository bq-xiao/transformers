from transformers import pipeline, T5Tokenizer, AutoModelForSeq2SeqLM

tokenizer = T5Tokenizer.from_pretrained("google/mt5-small", cache_dir="../download_model/mt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small", cache_dir="../download_model/mt5-small")
en_fr_translator = pipeline(task="translation_en_to_zh", model=model, tokenizer=tokenizer)
str = en_fr_translator("How old are you?")
print(str)