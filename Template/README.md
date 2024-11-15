```python
from model import LLM
model = LLM()

model.make_dataset() # traindataset을 만듬 (model.train_dataset으로 결과를 볼 수 있음)
model.view_data(idx) # idx에 해당하는 전처리된 데이터를 자연어의 형태로 확인
model.train() # 만든 traindataset으로 훈련
model.inferece() # 훈련된 모델을 바탕으로 추론
```

### Baseline과 똑같이 동작하는 Template
epoch은 2로 고정하고 모델 이름만 바꾸면서 어떤 모델이 좋을지를 비교해보도록 합시다.

#### 모델 후보
1. beomi/gemma-ko-2b (기본 모델)
2. meta-llama/Llama-3.1-8B-Instruct
3. meta-llama/Llama-3.2-3B-Instruct
4. beomi/Llama-3-Open-Ko-8B
5. Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M
6. Bllossom/llama-3.2-Korean-Bllossom-3B
뭐 더 있으면 추가해도 됨   

* argument model_name에 이름 말고 /data/ephemeral/home/code/outputs_gemma-ko-2b와 같이 경로를 넣으면 훈련된 모델 load 가능   
* 귀찮으면 model = LLM('/data/ephemeral/home/code/outputs_gemma-ko-2b/checkpoint-1827') 이런식으로 넣어도 로드 되게 해놨음   
* inference 과정에서 batch 연산으로 좀 더 빠르게 추론할 수 있도록 해놨음(배치 연산하면 추론이 이상하게 되는 것 같아 다시 되돌렸음)

