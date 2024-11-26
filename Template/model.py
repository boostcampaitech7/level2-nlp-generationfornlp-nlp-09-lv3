import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from arguments import model_args
from peft import LoraConfig
import pandas as pd
from datasets import Dataset
import numpy as np
import evaluate
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from ast import literal_eval
from tqdm import tqdm
from prompts import prompts

class LLM:
    def __init__(self, route = None, prompt = None):
        self.args = model_args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if route == None:
            route = self.args.model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            route,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            route,
            trust_remote_code=True,
        )
        if self.tokenizer.chat_template == None:
            self.tokenizer.chat_template = (
                        "{% if messages[0]['role'] == 'system' %}"
                            "{% set system_message = messages[0]['content'] %}"
                            "{% endif %}"
                            "{% if system_message is defined %}{{ system_message }}{% endif %}"
                            "{% for message in messages %}"
                            "{% set content = message['content'] %}"
                            "{% if message['role'] == 'user' %}"
                            "{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}"
                            "{% elif message['role'] == 'assistant' %}"
                            "{{ content + '<end_of_turn>\n' }}"
                            "{% endif %}"
                            "{% endfor %}"
                        )
            self.response_template = "<start_of_turn>model"
        else:
            self.response_template = "assistant<|end_header_id|>"
        
        
        self.PROMPT_NO_QUESTION_PLUS = """지문:
                        {paragraph}

                        질문:
                        {question}

                        선택지:
                        {choices}

                        1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
                        정답:"""

        self.PROMPT_QUESTION_PLUS = """지문:
                        {paragraph}

                        질문:
                        {question}

                        <보기>:
                        {question_plus}

                        선택지:
                        {choices}

                        1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
                        정답:"""
        self.prompt = prompts.baseline if prompt == None else prompt
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'right'
        self.acc_metric = evaluate.load("accuracy")
# ------------------------------------------------------------------------------------------------------------

    def make_dataset(self):
        #make_dataset 안에서만 쓸 함수 두개 지정 (매핑함수)
        def formatting_prompts_func(example):
            output_texts = []
            for i in range(len(example["messages"])):
                output_texts.append(
                    self.tokenizer.apply_chat_template(
                        example["messages"][i],
                        tokenize=False,
                    )
                )
            return output_texts

        def tokenize(element):
            outputs = self.tokenizer(
                formatting_prompts_func(element),
                truncation=False,
                padding=False,
                return_overflowing_tokens=False,
                return_length=False,
            )
            return {
                "input_ids": outputs["input_ids"],
                "attention_mask": outputs["attention_mask"],
            }

        dataset = pd.read_csv(self.args.data_route)

        # Flatten the JSON dataset
        records = []
        for _, row in dataset.iterrows():
            problems = literal_eval(row['problems'])
            record = {
                'id': row['id'],
                'paragraph': row['paragraph'],
                'question': problems['question'],
                'choices': problems['choices'],
                'answer': problems.get('answer', None),
                "question_plus": problems.get('question_plus', None),
            }
            # Include 'question_plus' if it exists
            if 'question_plus' in problems:
                record['question_plus'] = problems['question_plus']
            records.append(record)
                
        # Convert to DataFrame
        df = pd.DataFrame(records)
        dataset = Dataset.from_pandas(df)

        processed_dataset = []
        for i in range(len(dataset)):
            choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(dataset[i]["choices"])])

            # <보기>가 있을 때
            if dataset[i]["question_plus"]:
                user_message = self.PROMPT_QUESTION_PLUS.format(
                    paragraph=dataset[i]["paragraph"],
                    question=dataset[i]["question"],
                    question_plus=dataset[i]["question_plus"],
                    choices=choices_string,
                )
            # <보기>가 없을 때
            else:
                user_message = self.PROMPT_NO_QUESTION_PLUS.format(
                    paragraph=dataset[i]["paragraph"],
                    question=dataset[i]["question"],
                    choices=choices_string,
                )

            # chat message 형식으로 변환
            processed_dataset.append(
                {
                    "id": dataset[i]["id"],
                    "messages": [
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": f"{dataset[i]['answer']}"}
                    ],
                    "label": dataset[i]["answer"],
                }
            )
        processed_dataset = Dataset.from_pandas(pd.DataFrame(processed_dataset))

        tokenized_dataset = processed_dataset.map(
            tokenize,
            remove_columns=list(processed_dataset.features),
            batched=True,
            load_from_cache_file=True,
            desc="Tokenizing",
        )
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= 2048)  
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

        self.train_dataset = tokenized_dataset['train']
        self.eval_dataset = tokenized_dataset['test']
        train_dataset_token_lengths = [len(self.train_dataset[i]["input_ids"]) for i in range(len(self.train_dataset))]
        print(f"max token length: {max(train_dataset_token_lengths)}")
        print(f"min token length: {min(train_dataset_token_lengths)}")
        print(f"avg token length: {np.mean(train_dataset_token_lengths)}")

    def view_data(self,idx):
        print(self.tokenizer.decode(self.train_dataset['input_ids'][idx]))

# --------------------------------------------------------------------------------------------------------------------------

        # 모델의 logits 를 조정하여 정답 토큰 부분만 출력하도록 설정
    def preprocess_logits_for_metrics(self, logits, labels):
        logits = logits if not isinstance(logits, tuple) else logits[0]
        logit_idx = [self.tokenizer.vocab["1"],
                        self.tokenizer.vocab["2"],
                        self.tokenizer.vocab["3"],
                        self.tokenizer.vocab["4"], 
                        self.tokenizer.vocab["5"]]
        logits = logits[:, -2, logit_idx] # -2: answer token, -1: eos token
        return logits

    
        # metric 계산 함수
    def compute_metrics(self, evaluation_result):
        logits, labels = evaluation_result
        int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}


        # 토큰화된 레이블 디코딩
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
        labels = list(map(lambda x: int_output_map[x], labels))

        # 소프트맥스 함수를 사용하여 로그트 변환
        probs = torch.nn.functional.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)

        predictions = np.argmax(probs, axis=-1)

        # 정확도 계산
        acc = self.acc_metric.compute(predictions=predictions, references=labels)
        return acc
    
# --------------------------------------------------------------------------------------------------------------------------

    def train(self):
        print(self.response_template,'부터 로스를 계산합니다.')
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=self.response_template,
            tokenizer=self.tokenizer,
        )
        peft_config = LoraConfig(
                r=6,
                lora_alpha=8,
                lora_dropout=0.05,
                target_modules=['q_proj', 'k_proj'],
                bias="none",
                task_type="CAUSAL_LM",
            )

        sft_config = SFTConfig(
            do_train=True,
            do_eval=True,
            lr_scheduler_type="cosine",
            max_seq_length=self.args.max_seq_length,
            output_dir='outputs_'+self.args.model_name.split('/')[-1],
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            num_train_epochs=self.args.num_train_epochs,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            logging_steps=self.args.logging_steps,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=2,
            save_only_model=True,
            report_to="none",
            gradient_accumulation_steps=4
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            peft_config=peft_config,
            args=sft_config,
        )

        trainer.train()

    def inference(self, test_df = None, mode = 'logit_base'):
        if test_df == None:
            test_df = pd.read_csv(self.args.test_route)

        # Flatten the JSON dataset
        records = []
        for _, row in test_df.iterrows():
            problems = literal_eval(row['problems'])
            record = {
                'id': row['id'],
                'paragraph': row['paragraph'],
                'question': problems['question'],
                'choices': problems['choices'],
                'answer': problems.get('answer', None),
                "question_plus": problems.get('question_plus', None),
            }
            # Include 'question_plus' if it exists
            if 'question_plus' in problems:
                record['question_plus'] = problems['question_plus']
            records.append(record)
                
        # Convert to DataFrame
        test_df = pd.DataFrame(records)

        test_dataset = []
        for i, row in test_df.iterrows():
            choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
            len_choices = len(row["choices"])
            
            # <보기>가 있을 때
            if row["question_plus"]:
                user_message = self.PROMPT_QUESTION_PLUS.format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    question_plus=row["question_plus"],
                    choices=choices_string,
                )
            # <보기>가 없을 때
            else:
                user_message = self.PROMPT_NO_QUESTION_PLUS.format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    choices=choices_string,
                )

            test_dataset.append(
                {
                    "id": row["id"],
                    "messages": [
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "label": row["answer"],
                    "len_choices": len_choices,
                }
            )

        pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    
        infer_results = []
        if mode == 'logit_base':
            self.model.eval()
            with torch.inference_mode():
                tar_probs = []
                answers = []
                for data in tqdm(test_dataset):
                    _id = data["id"]
                    messages = data["messages"]
                    len_choices = data["len_choices"]

                    outputs = self.model(
                        self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        ).to("cuda")
                    )
                    
                    logits = outputs.logits[:, -1].flatten().cpu()
                    
                    target_logit_list = [logits[self.tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]
                    probs = (
                        torch.nn.functional.softmax(
                            torch.tensor(target_logit_list, dtype=torch.float32)
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
                    infer_results.append({"id": _id, "answer": predict_value})
                    tar_probs.append(probs)
                    answers.append(predict_value)

            self.results = pd.DataFrame(infer_results)
            test_df['probs'] = tar_probs
            test_df['answers'] = answers

            self.test_df = test_df

            pd.DataFrame(infer_results).to_csv('output_logit_base.csv')

        elif mode == 'generative_base':
            generated_infer_results = []
            self.model.eval()
            with torch.inference_mode():
                for idx, data in enumerate(tqdm(test_dataset)):
                    _id = data["id"]
                    messages = data["messages"]
                    len_choices = data["len_choices"]

                    # 텍스트 생성을 위한 입력 데이터 준비
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to("cuda")
                    self.inputs = inputs
                    # 모델을 이용한 텍스트 생성
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=2,  # 최대 생성 토큰 수
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    self.outputs = outputs
                    # 생성된 텍스트 디코딩
                    generated_text = self.tokenizer.batch_decode(
                        outputs[:, inputs.shape[1]], skip_special_tokens=True
                    )[0]
                    generated_text = generated_text.strip()
                    try:
                        generated_text = int(generated_text)
                    except:
                        print(generated_text,'에서 오류 발생')
                        generated_text = 1
                    # 생성된 텍스트와 라벨을 결과 리스트에 추가
                    generated_infer_results.append({
                        "id": _id,  # 고유 ID
                        "answer": int(generated_text),  # 생성된 텍스트
                        "label": data["label"]  # 실제 라벨이 있다면, data에서 가져옴
                    })
            generated_infer_results = pd.DataFrame(generated_infer_results)
            self.results = generated_infer_results
            generated_infer_results.to_csv('output_generative_base.csv')
            
