import ollama

# 모델 호출 예시
response = ollama.chat(model="aya-expanse:8b", messages=[{"role": "user", "content": "Hello, how are you?"}])
print(response)
