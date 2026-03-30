# -*- coding: utf-8 -*-
"""1. Install Dependencies"""

!pip install -q "unsloth[colab-new]" xformers transformers trl datasets

"""2. Imports and Setup"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth.chat_templates import train_on_responses_only
import os
import json

"""3. Load Base Model"""

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

"""4. Baseline Inference (Before Training)"""

print("===== BEFORE TRAINING =====")
FastLanguageModel.for_inference(model)

messages = [
    {
        "role": "user",
        "content": "A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?"
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=300,
    use_cache=False, # Changed from True to False
    temperature=0.7,
    top_p=0.9,
)

generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(response)

"""5. Load Dataset, Train / Validation Split"""

raw_dataset = load_dataset("donghuna/generated_code-gsm8k-plan", split="train")
print(raw_dataset.column_names)
print(raw_dataset[0])

# Shuffle once for reproducibility
raw_dataset = raw_dataset.shuffle(seed=42)

# Optional: make train/validation split
split_dataset = raw_dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

"""6. Add LoRA Adapters (QLoRA)"""

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    max_seq_length=max_seq_length,
)

"""7. Dataset Formatting"""

def formatting_prompts_func(examples):
    texts = []

    questions = examples["question"]
    answers = examples["answer"]
    solutions = examples["solution"]

    for q, a, s in zip(questions, answers, solutions):
        # Recommended: teach reasoning + final answer
        assistant_text = f"""Plan and solution:
{s}

Final Answer:
{a}"""

        convo = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": assistant_text},
        ]

        formatted_text = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(formatted_text)

    return {"text": texts}

train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

# Keep only text column
cols_to_remove_train = [c for c in train_dataset.column_names if c != "text"]
cols_to_remove_eval = [c for c in eval_dataset.column_names if c != "text"]

train_dataset = train_dataset.remove_columns(cols_to_remove_train)
eval_dataset = eval_dataset.remove_columns(cols_to_remove_eval)

print(train_dataset.column_names)
print(train_dataset[0]["text"])

"""8. Training Setup"""

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        output_dir="outputs",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    ),
)

# Train only on assistant responses
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# Debug one sample
i = 5
print("FULL INPUT:")
print(tokenizer.decode(trainer.train_dataset[i]["input_ids"]))

print("\nMASKED LABELS VIEW:")
space = tokenizer(" ", add_special_tokens=False)["input_ids"][0]
labels_view = [space if x == -100 else x for x in trainer.train_dataset[i]["labels"]]
print(tokenizer.decode(labels_view))

"""9. Train the Model"""

trainer_stats = trainer.train()

"""10. Inference After Training"""

print("===== AFTER TRAINING =====")
FastLanguageModel.for_inference(model)

messages = [
    {
        "role": "user",
        "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=300,
    use_cache=False,
    temperature=0.7,
    top_p=0.9,
)

generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(response)

"""11. Save Model"""

model.save_pretrained("lora_plan_model")
tokenizer.save_pretrained("lora_plan_model")