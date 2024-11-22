import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
import re
import ast  # JSON-like 문자열을 파싱하기 위해 사용

# 모델 로드 (Bllossom/llama-3.2-Korean-Bllossom-3B 모델)
model_name = 'Bllossom/llama-3.2-Korean-Bllossom-3B'  # Llama 모델 로드
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# 텍스트 클리닝 함수
def clean_text(text):
    # 대괄호와 따옴표 제거하고 앞뒤 공백 제거
    cleaned = re.sub(r'[\[\]"]', '', text).strip()
    return cleaned

# 번역 템플릿
def translate_text(text, target_language='en'):
    # 번역 프롬프트 생성
    prompt = f"Translate the following text to {target_language}: {text}"
    
    # 텍스트 토큰화
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # 모델로부터 번역 생성
    translated_tokens = model.generate(inputs.input_ids, max_length=512, num_beams=5, early_stopping=True)
    
    # 생성된 텍스트 디코딩
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# 데이터 읽기
data = pd.read_csv('./data/train.csv')

# Flatten된 데이터 생성
records = []
for _, row in data.iterrows():
    problems = ast.literal_eval(row['problems'])
    record = {
        'id': row['id'],
        'paragraph': row['paragraph'],
        'question': problems['question'],
        'choices': problems['choices'],
    }
    records.append(record)

# Flatten된 DataFrame 생성
flattened_data = pd.DataFrame(records)

# 영어 번역 작업 (질문)
translate_questions = []
for question in tqdm(flattened_data['question'], desc='영어 번역 중 (Question)...', total=len(flattened_data)):
    translated_question = translate_text(question, target_language='en')  # 영어로 번역
    translate_questions.append(clean_text(translated_question))
    print(translated_question)

flattened_data['question'] = translate_questions

# 영어 번역 작업 (선택지)
translate_choices = []
for choices in tqdm(flattened_data['choices'], desc='영어 번역 중 (Choices)...', total=len(flattened_data)):
    choices_text = " ".join(choices)  # 선택지를 하나의 문자열로 결합
    translated_choices = translate_text(choices_text, target_language='en')  # 영어로 번역
    translate_choices.append(clean_text(translated_choices))
    print(translated_choices)

flattened_data['choices'] = translate_choices

# 한국어 번역 작업 (영어로 번역된 질문)
translate_questions_ko = []
for question in tqdm(flattened_data['question'], desc='한국어 번역 중 (Question)...', total=len(flattened_data)):
    translated_question_ko = translate_text(question, target_language='ko')  # 한국어로 번역
    translate_questions_ko.append(clean_text(translated_question_ko))
    print(translated_question_ko)

flattened_data['question'] = translate_questions_ko

# 한국어 번역 작업 (영어로 번역된 선택지)
translate_choices_ko = []
for choices in tqdm(flattened_data['choices'], desc='한국어 번역 중 (Choices)...', total=len(flattened_data)):
    translated_choices_ko = translate_text(choices, target_language='ko')  # 한국어로 번역
    translate_choices_ko.append(clean_text(translated_choices_ko))
    print(translated_choices_ko)

flattened_data['choices'] = translate_choices_ko

# 결과 저장
flattened_data.to_csv('./data/train_3B_translated.csv', index=False)
