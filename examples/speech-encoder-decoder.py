from typing import Optional

import torch
import torchaudio
from datasets import load_dataset, Audio
from torch import nn

from transformers import AutoTokenizer, AutoFeatureExtractor, Wav2Vec2ForPreTraining, BartModel
from transformers import SpeechEncoderDecoderModel
import torchaudio.transforms as T

SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load("D:\\pyworkspace\\datasets\\downloads\\extracted\\9e3f31c39a3f8340eda00d4eda26d7bf5cd7f0a33a6af6e4ac0559e2c54cf80f\\en-US~JOINT_ACCOUNT\\602ba55abb1e6d0fbce92065.wav")
n_fft = 2048
win_length = None
hop_length = 512
n_mels = 256
n_mfcc = 256

mfcc_transform = T.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=n_mfcc,
    melkwargs={
        "n_fft": n_fft,
        "n_mels": n_mels,
        "hop_length": hop_length,
        "mel_scale": "htk",
    },
)

mfcc = mfcc_transform(SPEECH_WAVEFORM)
# a = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained("../download_model/wav2vec2-base", "../download_model/bart-base", )
# print(a)
LANG_ID = "en-US"

encoder_id = "facebook/wav2vec2-base"  # acoustic model encoder
decoder_id = "facebook/bart-base"  # text decoder
# 特征提取
feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id, cache_dir="../download_model/wav2vec2-base")
tokenizer = AutoTokenizer.from_pretrained(decoder_id, cache_dir="../download_model/bart-base")
dataset = load_dataset("PolyAI/minds14", LANG_ID, split="train", trust_remote_code=True,
                       cache_dir=r'D:\pyworkspace\datasets',
                       download_mode='reuse_cache_if_exists'
                       )
print(dataset[0])
class CustModel(nn.Module):
    def __init__(self, wav2Vec, base_model):
        super(CustModel, self).__init__()
        self.wav2Vec = wav2Vec
        self.base_model = base_model

    def forward(self,
                input_values: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                decoder_input_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                ):
        extract_features = self.wav2Vec.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self.wav2Vec._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.wav2Vec.feature_projection(extract_features)
        hidden_states = self.wav2Vec._mask_hidden_states(
            hidden_states, mask_time_indices=None, attention_mask=attention_mask
        )

        decoder_ouput = self.base_model(inputs = hidden_states,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels)
        return decoder_ouput

# Combine pre-trained encoder and pre-trained decoder to form a Seq2Seq model
#model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained("../download_model/wav2vec2-base", "../download_model/bert-base-uncased", )

#model.config.decoder_start_token_id = tokenizer.cls_token_id
#model.config.pad_token_id = tokenizer.pad_token_id

bart_model = BartModel.from_pretrained("../download_model/bart-base")
wav2vec_pre =  Wav2Vec2ForPreTraining.from_pretrained("../download_model/wav2vec2-base")
encoder = bart_model.encoder
decoder = bart_model.decoder
#config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
speech= SpeechEncoderDecoderModel(encoder=encoder,decoder=decoder)
speech.config.decoder_start_token_id = tokenizer.cls_token_id
speech.config.pad_token_id = tokenizer.pad_token_id
print(speech)
model = CustModel(wav2vec_pre.wav2vec2, speech)
print(model)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
print(dataset[0])

def prepare_dataset(batch):
    # audio_arrays = [x["array"] for x in examples["audio"]]
    # inputs = feature_extractor(
    #     audio_arrays,
    #     sampling_rate=16000,
    #     padding=True,
    #     max_length=100000,
    #     truncation=True,
    # )
    audio = batch["audio"]
    audio_arrays = [x["array"] for x in audio]
    texts = [x for x in batch["transcription"]]
    input_values = feature_extractor(audio_arrays, sampling_rate=audio[0]["sampling_rate"],
                                     #return_tensors='pt',
                                     max_length=100000, padding="max_length", return_tensors='pt', truncation=True,
                                     return_attention_mask = True
                                     )
    labels = tokenizer(texts,
                       #return_tensors='pt',
                       max_length=512, padding="max_length", return_tensors='pt', truncation=True
                       )
    #input_values["decoder_input_ids"] = labels["input_ids"]
    #input_values["decoder_attention_mask"] = labels["attention_mask"]
    input_values["labels"] = labels["input_ids"]
    return input_values

tokenized_datasets = dataset.map(prepare_dataset, batched=True,
                                 remove_columns= ["english_transcription", "intent_class", "lang_id"])
tokenized_datasets.set_format("torch")
tokenized_datasets = tokenized_datasets.train_test_split(train_size=0.8)

from transformers import TrainingArguments, Trainer, DefaultDataCollator

data_collator = DefaultDataCollator()
batch_size = 16
training_args = TrainingArguments(
    output_dir="speech-encoder-decoder",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=2000,
    gradient_checkpointing=False,
    fp16=True,
    group_by_length=True,
    eval_strategy="steps",
    per_device_eval_batch_size=batch_size,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    greater_is_better=False,
    push_to_hub=False,
    save_total_limit=2,
    report_to='tensorboard',
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=feature_extractor,
    data_collator=data_collator,
    #compute_metrics=compute_metrics,
)
print("start trainning ... ")
trainer.train()