# 힌트 생성
system_prompt = """
나는 대한민국 수능 전문가이다. 국어 또는 사회 영역의 지문과 질문이 주어졌을 때, 학생들이 문제를 잘 풀 수 있도록 힌트를 제공하고자 한다.
"""

user_prompt_no_question_plus = """
지문과 질문을 읽고 학생들에게 도움이 될 수 있는 힌트를 제공합니다.
힌트는 200자 이내의 한국어로 작성합니다.

### 문제
지문: {paragraph}
질문: {question}
선택지: {choices}
힌트:
"""

user_prompt_question_plus = """
지문과 질문을 읽고 학생들에게 도움이 될 수 있는 힌트를 제공합니다.
힌트는 200자 이내의 한국어로 작성합니다.

### 문제
지문: {paragraph}
질문: {question}
<보기>: {question_plus}
선택지: {choices}
힌트:
"""

# 힌트를 생성하는 함수 정의
def generate_hint(paragraph, question, choices, question_plus=None):
    # user_prompt 선택: question_plus가 있는 경우와 없는 경우를 분리
    if question_plus:
        user_prompt = user_prompt_question_plus
    else:
        user_prompt = user_prompt_no_question_plus

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt.format(
            paragraph=paragraph,
            question=question,
            choices=choices,
            question_plus=question_plus  # question_plus가 있을 때만 전달
        )}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# 힌트 생성 및 데이터셋 업데이트
hint_list = []
for i in tqdm(range(len(dataset))):
    paragraph = dataset[i]['paragraph']
    question = dataset[i]['question']
    choices = dataset[i]['choices']
    question_plus = dataset[i].get('question_plus', '')  # 질문에 추가적인 정보가 있는지 확인
    answer = dataset[i]['answer']

    # 힌트 생성
    hint = generate_hint(paragraph, question, choices, question_plus)

    # 생성된 힌트를 paragraph 앞에 붙이기
    updated_paragraph = hint + "\n" + paragraph
    dataset[i]['paragraph'] = updated_paragraph  # 힌트를 paragraph 앞에 추가

    hint_list.append(hint)

# 이제 reformat_test에 수정된 paragraph가 반영됩니다.
