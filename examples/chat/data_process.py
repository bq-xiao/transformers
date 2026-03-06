import logging as logger
import re

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

# 上面的配置类
from config import train_args
from transformers import AutoTokenizer, BertTokenizer


def get_dataset(source_dataset, tokenizer, args):
    """
    The format we need  is `<BOS><User>utterance1<EOS><Assistant>utterance2<EOS><User>utterance3<EOS><Assistant>utterance4<EOS>`
    """
    dialogues = []

    for example in tqdm(source_dataset["train"]):
        record = example["instruction"] + example["output"]
        utterances = re.split(r"(Human:|Assistant:)", record)

        utterances = [
            x.strip()
            for x in utterances
            if x.strip() not in ["Human:", "Assistant:", ""]
        ]
        dialogues.append(utterances)

    logger.info(f"There are {len(dialogues)} dialogues.")

    print(dialogues[0])

    conversation_list = []

    for utterances in tqdm(dialogues):
        # 每个对话以BOS开头
        input_ids = [args.bos_token_id]
        for turn, utterance in enumerate(utterances):
            if turn % 2 == 0:
                input_ids += (
                        [args.user_token_id]
                        + tokenizer.encode(utterance, add_special_tokens=False)
                        + [args.eos_token_id]
                )
            else:
                input_ids += (
                        [args.bot_token_id]
                        + tokenizer.encode(utterance, add_special_tokens=False)
                        + [args.eos_token_id]
                )
        # 不能超过model_max_length
        if len(input_ids) <= tokenizer.model_max_length:
            conversation_list.append(input_ids)

    tokenized_datasets = Dataset.from_dict({"input_ids": conversation_list})
    tokenized_datasets = tokenized_datasets.with_format("torch")
    # 数据集拆分
    train_valid = tokenized_datasets.train_test_split(test_size=args.valid_size)
    tokenized_datasets = DatasetDict(
        {
            "train": train_valid["train"],
            "valid": train_valid["test"],
        }
    )

    tokenized_datasets.save_to_disk(args.dataset_name)

    print(tokenized_datasets)

    return tokenized_datasets


if __name__ == "__main__":
    # 上面保存的分词器
    tokenizer = BertTokenizer.from_pretrained(r"nlp_gpt3_text-generation_chinese-base")
    special_tokens = {
        'additional_special_tokens': [
            '<Human>',
            '<Assistant>'
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    # 我们针对该数据集进行处理，转换成想要的格式
    source_dataset = load_dataset(r"multiturn_chat")
    # 获取数据集，并保存到磁盘
    get_dataset(source_dataset, tokenizer, train_args)
