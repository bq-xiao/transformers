import evaluate
import numpy as np
from datasets import Dataset
from transformers import BertTokenizer, BartForConditionalGeneration, DefaultDataCollator, TrainingArguments, \
    Trainer, EarlyStoppingCallback

tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese", cache_dir="../download_model/bart-base-chinese")
#print(tokenizer)

model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese", cache_dir="../download_model/bart-base-chinese")
#print(model.model)
#print(model.model.config)

file = r'news-commentary-v18.en-zh.tsv'
lines = open(file, encoding='utf-8').read().strip().split('\n')
# Split every line into pairs and normalize
rows = []
for l in lines:
    row = l.split('\t')
    if len(row[0]) == 0 or len(row[1]) == 0:
        continue
    rows.append({'input_text': row[0], 'target_text': row[1]})
rows = rows[:10]
dataset = Dataset.from_list(rows)

def preprocess_function(examples):
    inputs, targets = examples['input_text'], examples['target_text']

    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", return_tensors='pt', truncation=True)
    labels = tokenizer(targets, max_length=512, padding="max_length", return_tensors='pt', truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# translation_data = [
#     {'input_text': '这是一个关于自然语言处理的教程', 'target_text': 'This is a tutorial about Natural Language Processing'}
# ]

metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = preds.argmax(-1)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# map
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns= ['input_text','target_text'])
tokenized_datasets.set_format("torch")
tokenized_datasets = tokenized_datasets.train_test_split(train_size=0.8)

batch_size = 2
data_collator = DefaultDataCollator()
device = 'cuda'
model.to(device)

training_args = TrainingArguments(
    output_dir="bart_translation",
    do_eval=True,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end = True,
    #auto_find_batch_size = True,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    push_to_hub=False,
    report_to='tensorboard',
    logging_dir="logs",
    logging_strategy="steps",
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    processing_class=tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()