from dataclasses import dataclass, field
from typing import Optional


@dataclass
class prompts:
    baseline: Optional[str] = field(
        default = '지문을 읽고 질문의 답을 구하세요.',
        metadata = {'help' : "baseline"},
    )
    baseline_eng: Optional[str] = field(
        default = 'Read paragraph, and select only one answer between choices.',
        metadata = {'help' : 'baseline ENG ver'},
    )
    chainofthought: Optional[str] = field(
        default = '지문을 읽고 단계별로 생각하여 정답을 구하세요.',
        metadata = {'help' : "COT(Chain Of Thought)"},
    )
    chainofthought_eng: Optional[str] = field(
        default = 'As a smart student answer the given question.\
                    Read paragraph, and select only one answer between choices.',
        metadata = {'help' : "COT(Chain Of Thought) ENG ver"},
    )
    planing: Optional[str] = field(
     default = """Read the following passage and answer the given question. 
                    Explain each step in order while solving the problem. 
                    Passage: [Passage content] Question: [Question content] 
                    Step 1: Find the important information from the passage and identify the key elements required for the question.
                    Step 2: Compare the content mentioned in the passage with the question and extract relevant information.
                    Step 3: Provide the reasoning needed to derive the answer, and finally write the answer.""",
        metadata = {'help' : "Planing"},
    )
    manual: Optional[str] = field(
        default = '국어 시험 문제를 푸는 대한민국의 고3 수험생으로서 위의 요약을 바탕으로 다음 문제의 답을 구하세요.\
                    문제를 풀이할 때, 반드시 지문을 참고하세요.\
                    문제는 무조건 1개의 정답만 있습니다.\
                    문제를 풀이할 때 모든 선택지들을 검토하세요.\
                    모든 선택지마다 정답인 근거와 정답이 아닌 근거를 설명하세요.\
                    다음의 형식을 따라 답변하세요.\
                    최종 정답: (최종 정답)\
                    1번: (선택지 1번에 대한 답변) + "근거"\
                    2번: (선택지 2번에 대한 답변) + "근거"\
                    3번: (선택지 3번에 대한 답변) + "근거"\
                    4번: (선택지 4번에 대한 답변) + "근거"\
                    5번: (선택지 5번에 대한 답변) + "근거"',
        metadata = {'help' : "메뉴얼을 상세히 지정"}
    )
    cot_5step: Optional[str] = field(
        default = """1. Read and understand the paragraph meticulously.
                    2. Read and understand the question carefully.
                    3. Describe the process of solving the question based on your understanding.
                    4. What is the answer?
                    5. Choose the answer that most closely resembles the expected answer in the choice.""",
        metadata = {'help' : "cot_5step"},
    )
