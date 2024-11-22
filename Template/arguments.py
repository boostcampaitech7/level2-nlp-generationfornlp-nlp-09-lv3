from dataclasses import dataclass, field
from typing import Optional


@dataclass
class model_args:
    model_name: Optional[str] = field(
        default = 'CarrotAI/Llama-3.2-Rabbit-Ko-3B-Instruct',
        metadata = {'help' : "model_name"},
    )
    data_route: Optional[str] = field(
        default = '../data/train.csv',
        metadata = {'help' : "model_name"},
    )
    test_route: Optional[str] = field(
        default = '../data/test.csv',
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
        default = 6,
        metadata = {'help' : 'max_epoch'}
    )
    learning_rate: float = field(
        default = 2e-5,
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