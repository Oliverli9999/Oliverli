# finetune_gpt2_squad.py

import os
os.environ["WANDB_DISABLED"] = "true"   

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

MODEL_NAME = "openai-community/gpt2"


OUTPUT_DIR = "app/gpt2_finetuned"



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)



dataset = load_dataset("rajpurkar/squad")


MAX_TRAIN_SAMPLES = 1000
train_raw = dataset["train"].select(range(min(MAX_TRAIN_SAMPLES, len(dataset["train"]))))



PREFIX = "That is a great question. "
SUFFIX = " Let me know if you have any other questions."

def format_example(example):
    context = example["context"]
    question = example["question"]
    answers = example["answers"]["text"]
    answer = answers[0] if len(answers) > 0 else "I am not sure."

    text = (
        "Context: " + context + "\n"
        "Question: " + question + "\n"
        "Answer: " + PREFIX + answer + SUFFIX
    )
    return {"text": text}

train_formatted = train_raw.map(format_example, remove_columns=train_raw.column_names)


# ---------- 4. Tokenize ----------
def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )

train_tokenized = train_formatted.map(tokenize_function, batched=True)
train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,          
)



training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=1,          
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=500,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    data_collator=data_collator,
)



if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Finished fine-tuning! Model saved to: {OUTPUT_DIR}")
