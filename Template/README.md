```python
from model import LLM
model = LLM()
model = LLM('outputs_gemma-ko-2b/checkpoint-1827') # 훈련된 모델을 가져올 수 있음
model.make_dataset() # traindataset을 만듬 (model.train_dataset으로 결과를 볼 수 있음)
model.view_data(idx) # idx에 해당하는 전처리된 데이터를 자연어의 형태로 확인
model.train() # 만든 traindataset으로 훈련
model.inference() # 훈련된 모델을 바탕으로 추론 / 기본값 : logit_base
model.inference(mode = 'generative_base') # 생성 방식으로 추론 / 간혹 생성이 잘못된건 1로 내보냄
```   


```python
from model import LLM
from prompts import user_prompts, system_prompts
system = system_prompts.baseline
user = user_prompts.klue_hint
user_plus = user_prompts.klue_hint_plus
model = LLM( # route = 'outputs_gemma-ko-2b/checkpoint-1827'
            system_prompt=system,
             user_prompt=user,
             user_prompt_plus=user_plus)
model.make_dataset()
    # 이렇게 하면 프롬프트별로 eval 결과를 볼 수 있음.
    # 다만 결과는 model이 다시 정의돼서 날라가긴 함
    # 나중에 결과 기록할수있게 바꿔두겠음.

```

### 프롬프트 팁
1. system prompt는 간단하게 like '다음 지문을 읽고 정답을 고르세요.'
2. user prompt는 짧고 간결하면서도 지시를 명확하게
>다음 지문을 읽고 정답을 하나만 고르세요.
 정답을 고르기 어렵다면 힌트를 참고하세요.
 출력 형태 : [정답에 대한 근거] # [정답 번호]
 
### 주의사항
model.inference(mode = 'generative_base')만 사용할 것
inference 후 model.results를 치면 generate 된 결과물이 나옴.
결과물을 보고 알아서 정답 번호를 유추할 것.

#### 예시
```python
submission = model.results
submission['answer'] = submission['answer'].apply(lambda x: x.split('#')[-1])
# 근거 # 정답번호 로 출력하도록 했으니 #으로 split하면 [근거, 정답번호]가 되고 그 리스트 중 제일 뒤에거(정답번호)
```
generative base인 만큼 저렇게 해도 예외가 분명 있음. 예외를 submission.fillna(1) 이런식으로 채우던가 하면 됨.