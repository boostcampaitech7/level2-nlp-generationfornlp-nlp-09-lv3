from model import LLM
'''
model.inference(mode = ‘generative_base’)로하면 생성기반 추론
model.inference(mode = ‘logit_base’)하면 로짓기반추론
model = LLM(prompt = 원하는프롬프트)하면 프롬프트 원하는걸로되고
model = LLM()으로하면 baseline프롬프트임
'''

model = LLM(prompt='baseline_eng')

model.make_dataset() # traindataset을 만듬 (model.train_dataset으로 결과를 볼 수 있음)
# model.view_data(idx) # idx에 해당하는 전처리된 데이터를 자연어의 형태로 확인
model.train() # 만든 traindataset으로 훈련
model.inference(mode='logit_base') # 훈련된 모델을 바탕으로 추론