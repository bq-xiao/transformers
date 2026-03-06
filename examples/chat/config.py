from dataclasses import dataclass

@dataclass
class TrainArguments:
    batch_size: int = 4
    weight_decay: float = 1e-1
    epochs: int = 10
    warmup_steps: int = 4000
    learning_rate: float = 5e-5 # 似乎还可以大一点
    logging_steps = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 2.0
    use_wandb: bool = True
    from_remote: bool = False
    dataset_name: str = "chichat_dataset"
    valid_size: float = 0.05
    model_name: str = "gpt2-chatbot-chinese"
    tokenizer_name: str = "gpt2-chatbot-chinese"
    device: str = 0
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    bot_token: str = "<Assistant>"
    user_token: str = "<User>"
    bos_token_id: int = 21128
    eos_token_id: int = 21129
    bot_token_id: int = 21130
    user_token_id: int = 21131
    ignore_index: int = -100


train_args = TrainArguments()