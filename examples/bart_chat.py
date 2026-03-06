import numpy as np
from datasets import load_dataset, DownloadMode

from transformers import BertTokenizer, BartForConditionalGeneration, EarlyStoppingCallback
from transformers import DefaultDataCollator
from transformers import TrainingArguments, Trainer

#tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese", cache_dir="download_model/bart-base-chinese")
tokenizer = BertTokenizer.from_pretrained("simple_qa_model/checkpoint-375")
print(tokenizer)
#model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese", cache_dir="download_model/bart-base-chinese")
model = BartForConditionalGeneration.from_pretrained("simple_qa_model/checkpoint-375")
print(model)

# dataset
simple_dataset = load_dataset("OpenStellarTeam/Chinese-SimpleQA", cache_dir="dataset/qa/Chinese-SimpleQA", download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

# preprocess
def preprocess_function(examples):
    inputs = examples['question']
    targets = examples['answer']
    model_inputs = tokenizer(inputs, max_length=256, padding="max_length", truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True)


    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

metric = None

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    # decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    # decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result

# map
tokenized_datasets = simple_dataset.map(preprocess_function, batched=True, remove_columns=simple_dataset["train"].column_names)
tokenized_datasets.set_format("torch")
tokenized_datasets = tokenized_datasets['train'].train_test_split(train_size=0.8)

batch_size = 8
data_collator = DefaultDataCollator()
device = 'cuda'
model.to(device)

training_args = TrainingArguments(
    output_dir="simple_qa_model",
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