from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

ds = load_dataset("BoostedJonP/JeromePowell-SFT")

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
BITS_AND_BYTES_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "float16",
}

bnb = BitsAndBytesConfig(**BITS_AND_BYTES_CONFIG)

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map="auto",
)

lora = LoraConfig(
    r=8,  # Low rank - keeps memory low
    lora_alpha=16,  # Scaling factor (2x rank)
    lora_dropout=0.1,  # Protect from overfitting and training imbalance
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Attention + MLP layers
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora)
model.print_trainable_parameters()


cfg = SFTConfig(
    output_dir="powell-phi3-lora",
    max_length=1536,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=6,
    num_train_epochs=3,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=False,
    fp16=True,
    packing=False,
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    peft_config=lora,
    train_dataset=ds["train"],
    formatting_func=lambda ex: tok.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are Jerome Powell, the Chairman of the Federal Reserve.",
            },
            {"role": "user", "content": ex["instruction"]},
            {"role": "assistant", "content": ex["output"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    ),
    args=cfg,
)


class PrintLossCallback(TrainerCallback):
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if logs is not None and "loss" in logs:
            print(f"Step {state.global_step}: Loss = {logs['loss']:.4f}")


trainer.add_callback(PrintLossCallback())

trainer.train()
