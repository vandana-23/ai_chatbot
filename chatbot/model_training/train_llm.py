# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
# from pathlib import Path

# #  Model and paths
# MODEL_NAME = "google/flan-t5-base"

# DATA_DIR = Path(__file__).resolve().parents[1] / "data"
# OUT_DIR = Path(__file__).resolve().parent / "fine_tuned_model"

# data_path = DATA_DIR / "company_data.txt"
# assert data_path.exists(), f"Dataset file not found: {data_path}"

# # Load dataset
# dataset = load_dataset("text", data_files=str(data_path))
# print("Dataset loaded successfully:", dataset)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# def tokenize(examples):
#     inputs = tokenizer(examples["text"], truncation=True)
#     # Self-supervised: use same text as target
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(examples["text"], truncation=True)
#     inputs["labels"] = labels["input_ids"]
#     return inputs

# # args = TrainingArguments(
# #     output_dir=str(OUT_DIR),
# #     per_device_train_batch_size=1,    
# #     gradient_accumulation_steps=2,    
# #     num_train_epochs=5,              
# #     logging_steps=5,
# #     save_strategy="no",               
# #     fp16=False,
# #     remove_unused_columns=False,
# #     report_to="none",
# # )


# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=dataset["train"].map(tokenize, batched=True, remove_columns=["text"]),
#     # tokenizer=tokenizer
#     processing_class=tokenizer,
# )

# trainer.train()

# model.save_pretrained(str(OUT_DIR))
# tokenizer.save_pretrained(str(OUT_DIR))
# print(f" Fine-tuned model saved to {OUT_DIR}")








from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, pipeline
from pathlib import Path

MODEL_NAME = "google/flan-t5-base"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parent / "fine_tuned_model"

data_path = DATA_DIR / "company_data.txt"
assert data_path.exists(), f"Dataset file not found: {data_path}"

dataset = load_dataset("text", data_files=str(data_path))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def tokenize(examples):
    inputs = [f"Instruction: {x}" for x in examples["text"]]
    targets = [f"Response: {x}" for x in examples["text"]]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=256)
    labels = tokenizer(targets, truncation=True, padding="max_length", max_length=256)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

args = TrainingArguments(
    output_dir=str(OUT_DIR),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    remove_unused_columns=False,
    learning_rate=3e-5,
    warmup_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"].map(tokenize, batched=True, remove_columns=["text"]),
)

trainer.train()
model.save_pretrained(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))

print("Fine-tuned model saved at", OUT_DIR)