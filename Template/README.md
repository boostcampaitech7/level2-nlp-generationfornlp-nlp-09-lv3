```python
from model import LLM
model = LLM()
model = LLM('outputs_gemma-ko-2b/checkpoint-1827') # 훈련된 모델을 가져올 수 있음
model = LLM(prompt = prompt) 프롬프트를 지정하여 훈련 / 추론할 수 있음
model.make_dataset() # traindataset을 만듬 (model.train_dataset으로 결과를 볼 수 있음)
model.view_data(idx) # idx에 해당하는 전처리된 데이터를 자연어의 형태로 확인
model.train() # 만든 traindataset으로 훈련
model.inference() # 훈련된 모델을 바탕으로 추론 / 기본값 : logit_base
model.inference(mode = 'generative_base') # 생성 방식으로 추론 / 간혹 생성이 잘못된건 1로 내보냄
```   

프롬프트를 다양하게 쓰고싶다면 ..   

```python
from prompts.py import prompts
import gc
import torch
prompts = [prompts.baseline, prompts.baseline_eng, prompts.chainofthought,
           prompts.chainofthought_eng, prompts.manual, prompts.planing]

for prompt in prompts:
    model = LLM(prompt = prompt)
    model.make_dataset()
    model.train()
    gc.collect()
    torch.cuda.empty_cache()
    # 이렇게 하면 프롬프트별로 eval 결과를 볼 수 있음.
    # 다만 결과는 model이 다시 정의돼서 날라가긴 함
    # 나중에 결과 기록할수있게 바꿔두겠음.
```

### Baseline과 똑같이 동작하는 Template
epoch은 2로 고정하고 모델 이름만 바꾸면서 어떤 모델이 좋을지를 비교해보도록 합시다
모델 후보
1. beomi/gemma-ko-2b (기본 모델)
2. meta-llama/Llama-3.1-8B-Instruct
3. meta-llama/Llama-3.2-3B-Instruct
4. beomi/Llama-3-Open-Ko-8B
5. Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M
6. Bllossom/llama-3.2-Korean-Bllossom-3B
뭐 더 있으면 추가해도 됨
