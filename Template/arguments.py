from dataclasses import dataclass, field
from typing import Optional


@dataclass
class model_args:
    model_name: Optional[str] = field(
        default = 'sh2orc/Llama-3.1-Korean-8B-Instruct',
        metadata = {'help' : "model_name"},
    )
    data_route: Optional[str] = field(
        default = '/data/ephemeral/home/code/datas/train.csv',
        metadata = {'help' : "model_name"},
    )
    test_route: Optional[str] = field(
        default = 'datas/test.csv',
        metadata = {'help' : 'test_route'}
    )
    max_seq_length: int = field(
        default = 2048,
        metadata = {'help' : 'max_length'}
    )
    per_device_train_batch_size: int = field(
        default = 1,
        metadata = {'help' : 'train_batch_size'}
    )
    per_device_eval_batch_size: int = field(
        default = 1,
        metadata = {'help' : 'eval_batch_size'}
    )
    num_train_epochs: int = field(
        default = 3,
        metadata = {'help' : 'max_epoch'}
    )
    learning_rate: float = field(
        default = 1e-5,
        metadata = {'help' : 'learning_rate'}
    )
    weight_decay: float = field(
        default = 0.01,
        metadata = {'help' : 'weight_decay'}
    )
    logging_steps : int = field(
        default = 1,
        metadata = {'help' : 'logging_steps'}
    )
    gradient_accumulation_steps : int = field(
        default = 8,
        metadata = {'help' : 'gradient_accumulation_steps'}
    )
    gradient_checkpointing : bool = field(
        default = True,
        metadata = {'help' : """gradient checkpointing을 True로 줘야 8b모델이 돌아감
                    대신 매우 느리니 주의"""}
    )

@dataclass
class BM25_retrieval_arguments:
    data_route: Optional[str] = field(
        default='/data/ephemeral/home/code/datas/train+generated.csv',
        metadata={'help': "데이터셋 위치입니다."}
    )
    test_data_route: Optional[str] = field(
        default='/data/ephemeral/home/level2-mrc-nlp-10/data/test_dataset',
        metadata={'help': '테스트 데이터셋 위치입니다.'}
    )
    k: int = field(
        default = 30,
        metadata={'help': '비슷한 문서 중 몇 개를 내보낼지를 결정합니다.'}
    )
    wiki_route: str = field(
        default = '/data/ephemeral/home/code/datas/wikipedia_documents.json',
        metadata={'help': '위키 데이터의 경로입니다.'}
    )
    data_path: str = field(
        default='./bm25_retrieval_result',
        metadata={'help': 'BM25 검색 결과를 저장할 경로입니다.'}
    )
    bm25_tokenizer: str = field(
        default="beomi/gemma-ko-2b",
        metadata={'help': 'BM25 검색에서 사용할 토크나이저를 설정합니다.'}
    )
    model_name: str = field(
        default="beomi/gemma-ko-2b",
        metadata={'help': '토크나이저를 지정합니다. BM25에서도 동일하게 사용할 수 있습니다.'}
    )