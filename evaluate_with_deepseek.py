# -*- coding: utf-8 -*-
"""12. LLM-as-a-Judge Evaluation (DeepSeek)"""

!pip install -q openai
from openai import OpenAI
from google.colab import userdata

os.environ["DEEPSEEK_API_KEY"] = userdata.get('api_key')

client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator for math solutions designed for high school students.

Evaluate the model answer based on:

1. correctness_score (1-5)
2. reasoning_score (1-5)
3. clarity_score (1-5)
4. student_friendliness_score (1-5)
5. final_verdict ("pass" or "fail")

Return ONLY JSON in this format:

{
  "correctness_score": 0,
  "reasoning_score": 0,
  "clarity_score": 0,
  "student_friendliness_score": 0,
  "final_verdict": "pass or fail",
  "short_reason": "brief explanation"
}
"""

def judge_with_deepseek(question, gold_answer, model_prediction):

    prompt = f"""
Evaluate the following math solution.

Question:
{question}

Reference Answer:
{gold_answer}

Model Answer:
{model_prediction}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    content = response.choices[0].message.content

    # Strip markdown code block fences if present
    if content.startswith('```json') and content.endswith('```'):
        content = content[len('```json'):-len('```')].strip()

    try:
        return json.loads(content)
    except:
        return {"error": "Invalid JSON", "raw_output": content}

question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
gold_answer = "#### 72"
prediction = "She sold 48 + 24 = 72 clips."

result = judge_with_deepseek(question, gold_answer, prediction)

print(result)

def generate_response(model, tokenizer, question):
    messages = [
        {
            "role": "user",
            "content": question
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
    return response.strip()

prediction = generate_response(model, tokenizer, question)

judge_result = judge_with_deepseek(
    question,
    gold_answer,
    prediction
)

print("Prediction:")
print(prediction)

print("\nJudge result:")
print(judge_result)

eval_subset = raw_dataset.select(range(5))

results = []

for sample in eval_subset:
    question = sample["question"]
    gold = sample["answer"]

    pred = generate_response(model, tokenizer, question)

    judge = judge_with_deepseek(question, gold, pred)

    results.append({
        "question": question,
        "prediction": pred,
        "judge": judge
    })

for i, r in enumerate(results):
    print(f"\n=== Sample {i+1} ===")
    print("Question:", r["question"])
    print("Prediction:", r["prediction"])
    print("Judge:", r["judge"])

def average(scores):
    return sum(scores) / len(scores) if scores else 0

summary = {
    "num_samples": len(results),
    "avg_correctness_score": average([r["judge"]["correctness_score"] for r in results if "correctness_score" in r["judge"]]),
    "avg_reasoning_score": average([r["judge"]["reasoning_score"] for r in results if "reasoning_score" in r["judge"]]),
    "avg_clarity_score": average([r["judge"]["clarity_score"] for r in results if "clarity_score" in r["judge"]]),
    "avg_student_friendliness_score": average([r["judge"]["student_friendliness_score"] for r in results if "student_friendliness_score" in r["judge"]]),
    "pass_rate": sum(r["judge"].get("final_verdict") == "pass" for r in results) / len(results),
}

print(summary)

with open("deepseek_judge_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)