import torch
import pandas as pd
import numpy as np
from ast import literal_eval
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# ------------------- 1. Configurations -------------------
# 학습된 Checkpoint 경로
CHECKPOINT_PATH = "outputs_gemma/checkpoint-4491"

# Test 데이터 경로
TEST_DATA_PATH = "test.csv"

# 결과 저장 경로
OUTPUT_PATH = "output.csv"

# ------------------- 2. 모델 및 토크나이저 로드 -------------------
model = AutoPeftModelForCausalLM.from_pretrained(
    CHECKPOINT_PATH,
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    CHECKPOINT_PATH,
    trust_remote_code=True,
)

# ------------------- 3. 데이터 로드 및 전처리 -------------------
def load_test_data(file_path):
    """CSV 파일 로드 및 JSON 구조 Flatten"""
    df = pd.read_csv(file_path)
    records = []

    for _, row in df.iterrows():
        problems = literal_eval(row["problems"])
        records.append({
            "id": row["id"],
            "paragraph": row["paragraph"],
            "question": problems["question"],
            "choices": problems["choices"],
            "answer": problems.get("answer", None),
            "question_plus": problems.get("question_plus", None),
        })

    return pd.DataFrame(records)

test_df = load_test_data(TEST_DATA_PATH)

# ------------------- 4. 프롬프트 생성 -------------------
PROMPT_TEMPLATES = {
    "with_question_plus": """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:""",
    "no_question_plus": """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:""",
}

def create_prompt(row):
    """프롬프트 생성 함수"""
    choices_str = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
    template_key = "with_question_plus" if row["question_plus"] else "no_question_plus"
    return PROMPT_TEMPLATES[template_key].format(
        paragraph=row["paragraph"],
        question=row["question"],
        question_plus=row.get("question_plus", ""),
        choices=choices_str,
    )

def preprocess_test_data(df):
    """Test 데이터셋 전처리 및 메시지 생성"""
    dataset = []
    for _, row in df.iterrows():
        user_message = create_prompt(row)
        dataset.append({
            "id": row["id"],
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
            ],
            "label": row["answer"],
            "len_choices": len(row["choices"]),
        })
    return dataset

test_dataset = preprocess_test_data(test_df)

# ------------------- 5. 추론 -------------------
def infer(model, tokenizer, test_dataset, device="cuda"):
    """모델 추론 함수"""
    model.eval()
    pred_choices_map = {i: str(i + 1) for i in range(5)}  # 0-indexed to 1-indexed
    results = []

    with torch.inference_mode():
        for data in tqdm(test_dataset, desc="Running Inference"):
            messages = data["messages"]
            len_choices = data["len_choices"]

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits[:, -1].flatten().cpu()

            # 각 선택지의 확률 계산
            target_logits = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]
            probs = torch.nn.functional.softmax(torch.tensor(target_logits), dim=-1).numpy()

            # 예측 결과 저장
            predicted_answer = pred_choices_map[np.argmax(probs)]
            results.append({"id": data["id"], "answer": predicted_answer})

    return results

infer_results = infer(model, tokenizer, test_dataset)

# ------------------- 6. 결과 저장 -------------------
pd.DataFrame(infer_results).to_csv(OUTPUT_PATH, index=False)
