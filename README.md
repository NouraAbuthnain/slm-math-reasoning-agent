# SLM Math Reasoning Agent

> This project aims to make small language models useful as step-by-step math tutors for high school students.

This project explores fine-tuning a **Small Language Model (SLM)** for structured reasoning using **code generation**, with a focus on helping **high school students** solve math problems step-by-step.

The system is designed to provide:
- clear explanations  
- structured reasoning  
- reliable, verifiable answers  
 
## Project Goal

The main objective is to:

- Study how different fine-tuning strategies affect reasoning capabilities in SLMs  
- Transform the trained model into a **functional reasoning agent**

Additionally, the system is designed to support **high school students** by:

- Breaking down problems into step-by-step solutions  
- Generating clear and structured explanations  
- Using code to verify calculations  
- Encouraging logical thinking instead of just giving answers  

## Model

- **Model:** Qwen2.5-1.5B-Instruct  
- Lightweight instruction-tuned model suitable for SLM experiments  
- Fine-tuned using **QLoRA (4-bit quantization)** for efficiency  
  
## Dataset

[generated_code-gsm8k-plan](https://huggingface.co/datasets/donghuna/generated_code-gsm8k-plan) 

### Dataset Features

Each sample includes:

- Natural language math question  
- Step-by-step reasoning (plan)  
- Python code solution  
- Final numeric answer  

## Training Approach

The model is fine-tuned using:

- **Unsloth** for efficient training  
- **QLoRA** for memory-efficient adaptation  
- Supervised fine-tuning (SFT) on:
  - reasoning plans  
  - code generation  
  - final answers  

## Evaluation (LLM-as-a-Judge)

The model is evaluated using **DeepSeek API** as an external judge.

Evaluation criteria include:

- correctness  
- reasoning quality  
- clarity  
- student-friendliness  

This allows evaluation beyond simple exact-match accuracy.

## Agent System (LangChain + Pydantic)

The fine-tuned model is extended into an **agent** using LangChain.

### Agent Workflow

1. Generate a **plan**
2. Generate **Python code**
3. Execute the code
4. Produce a **final answer**

### State Management

The agent uses **Pydantic** to manage structured state:

- question  
- plan  
- generated code  
- execution result  
- final answer  

### Benefits

- Reliable numerical answers (via code execution)  
- Explainable reasoning  
- Structured problem-solving process  

## Target Use Case: High School Students

This project is designed for educational applications.

The system can:

- Help students understand math problems step-by-step  
- Provide structured solutions (plan → code → answer)  
- Verify results using code  
- Encourage deeper understanding instead of memorization  

### Potential Applications

- Homework assistance  
- Study companion  
- Math tutoring tool  

## Tech Stack

- **Unsloth** — efficient fine-tuning  
- **Transformers / PEFT** — model adaptation  
- **TRL (SFTTrainer)** — supervised fine-tuning  
- **LangChain / LangGraph** — agent framework  
- **Pydantic** — state management  
- **DeepSeek API** — LLM-as-a-judge  
