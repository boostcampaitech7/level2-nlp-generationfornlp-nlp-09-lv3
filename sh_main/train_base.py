import torch
import transformers
from ast import literal_eval
from datasets import Dataset
import pandas as pd
import random
import numpy as np
import evaluate
from tqdm import tqdm
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

pd.set_option('display.max_columns', None)

# ------------------- 1. 난수 고정 -------------------
def set_seed(random_seed=42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed()  # 난수 고정

# ------------------- 2. 데이터 로드 및 전처리 -------------------
def load_and_preprocess_dataset(file_path):
    # CSV 파일 로드
    dataset = pd.read_csv(file_path)
    records = []
    for _, row in dataset.iterrows():
        problems = literal_eval(row['problems'])
        record = {
            'id': row['id'],
            'paragraph': row['paragraph'],
            'question': problems['question'],
            'choices': problems['choices'],
            'answer': problems.get('answer'),
            'question_plus': problems.get('question_plus', ''),
        }
        records.append(record)

    # DataFrame 변환
    df = pd.DataFrame(records)
    df['full_question'] = df['question'] + ' ' + df['question_plus']
    df['question_length'] = df['full_question'].str.len()
    return df

# 데이터 로드
df = load_and_preprocess_dataset("train.csv")
dataset = Dataset.from_pandas(df)

# ------------------- 3. 모델 및 토크나이저 로드 -------------------
def load_model_and_tokenizer(model_name="beomi/gemma-ko-2b"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.chat_template = """{% if messages[0]['role'] == 'system' %}\
    {% set system_message = messages[0]['content'] %}{% endif %}\
    {% if system_message is defined %}{{ system_message }}{% endif %}\
    {% for message in messages %}{% set content = message['content'] %}\
    {% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}\
    {% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}\
    {% endif %}{% endfor %}"""
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# ------------------- 4. 데이터 변환 -------------------
PROMPT_TEMPLATES = {
    "no_question_plus": """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:""",
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
}

def create_prompt(data):
    choices_str = "\n".join(
        [f"{idx + 1} - {choice}" for idx, choice in enumerate(data["choices"])]
    )
    template = PROMPT_TEMPLATES[
        "with_question_plus" if data["question_plus"] else "no_question_plus"
    ]
    return template.format(
        paragraph=data["paragraph"],
        question=data["question"],
        question_plus=data.get("question_plus", ""),
        choices=choices_str,
    )

def preprocess_dataset(dataset):
    processed_data = []
    for i in tqdm(range(len(dataset)), desc="Processing Dataset"):
        data = dataset[i]
        prompt = create_prompt(data)
        processed_data.append({
            "id": data["id"],
            "messages": [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": str(data["answer"])},
            ],
            "label": data["answer"],
        })
    return Dataset.from_pandas(pd.DataFrame(processed_data))

processed_dataset = preprocess_dataset(dataset)

# ------------------- 5. 토큰화 -------------------
def tokenize_data(dataset, tokenizer):
    def formatting_func(example):
        return [
            tokenizer.apply_chat_template(message, tokenize=False)
            for message in example["messages"]
        ]
    
    def tokenize_func(element):
        outputs = tokenizer(
            formatting_func(element),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }
    
    return dataset.map(
        tokenize_func,
        remove_columns=list(dataset.features),
        batched=True,
        num_proc=4,
        desc="Tokenizing Dataset",
    )

tokenized_dataset = tokenize_data(processed_dataset, tokenizer)
tokenized_dataset = tokenized_dataset.filter(
    lambda x: len(x["input_ids"]) <= 1024, desc="Filtering Long Sequences"
)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

# ------------------- 6. Trainer 설정 및 학습 -------------------
peft_config = LoraConfig(
    r=6, lora_alpha=8, lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj'], bias="none", task_type="CAUSAL_LM",
)

data_collator = DataCollatorForCompletionOnlyLM(
    response_template="<start_of_turn>model", tokenizer=tokenizer
)

sft_config = SFTConfig(
    do_train=True, do_eval=True, lr_scheduler_type="cosine",
    max_seq_length=1024, output_dir="outputs_gemma",
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    num_train_epochs=3, learning_rate=2e-5, weight_decay=0.01,
    logging_steps=1, save_strategy="epoch", eval_strategy="epoch",
    save_total_limit=2, save_only_model=True, report_to="none",
)

# metric 계산 함수
def compute_metrics(evaluation_result):
    # Accuracy metric 로드
    acc_metric = evaluate.load("accuracy")
    
    logits, labels = evaluation_result

    # 토큰화된 레이블 디코딩
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
    
    # 정답 토큰 매핑
    int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
    labels = list(map(lambda x: int_output_map[x], labels))

    # 소프트맥스 함수를 사용하여 로그트 변환
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    predictions = np.argmax(probs, axis=-1)

    # 정확도 계산
    acc = acc_metric.compute(predictions=predictions, references=labels)
    return acc


trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=lambda logits, _: logits[:, -2],
    peft_config=peft_config,
    args=sft_config,
)

trainer.train()



