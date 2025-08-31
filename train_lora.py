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
from peft import LoraConfig
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
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)


cfg = SFTConfig(
    output_dir="powell-phi3-lora",
    max_length=1536,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=6,
    num_train_epochs=3,
    learning_rate=1.5e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=False,
    fp16=True,
    packing=False,
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
)

trainer = SFTTrainer(
    model=model,
    peft_config=lora,
    train_dataset=ds["train"],
    formatting_func=lambda ex: tok.apply_chat_template(
        [
            {
                "role": "user",
                "content": ex["instruction"] + ("\n\n" + ex["input"] if ex.get("input") else ""),
            },
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
