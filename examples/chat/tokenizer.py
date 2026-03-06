from config import train_args
from transformers import BertTokenizer


def save_tokenizer(
        vocab_path,
        model_name="gpt2-chatbot-chinese",
        bos_token="<BOS>",
        eos_token="<EOS>",
        bot_token="<Assistant>",
        user_token="<User>",
):
    #tokenizer = BertTokenizerFast(vocab_file=vocab_path, model_max_length=1024)
    tokenizer = BertTokenizer.from_pretrained(r"../../download_model/gpt2-distil-chinese-cluecorpussmall", model_max_length=1024)
    special_tokens = {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "additional_special_tokens": [bot_token, user_token],
    }
    tokenizer.add_special_tokens(special_tokens)

    tokenizer.save_pretrained(model_name)

save_tokenizer(
    "./vocab.txt",
    model_name=train_args.tokenizer_name,
    bos_token=train_args.bos_token,
    eos_token=train_args.eos_token,
    bot_token=train_args.bot_token,
    user_token=train_args.user_token,
)

