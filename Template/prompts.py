from dataclasses import dataclass, field
from typing import Optional


@dataclass
class system_prompts:
    baseline: Optional[str] = field(
        default = '지문을 읽고 질문의 답을 구하세요.',
        metadata = {'help' : "baseline"},
    )
    baseline_eng: Optional[str] = field(
        default = 'Read paragraph, and select only one answer between choices.',
        metadata = {'help' : 'baseline ENG ver'},
    )
    cot_easy: Optional[str] = field(
        default = '지문을 읽고 단계별로 생각하여 정답을 구하세요.',
        metadata = {'help' : "COT(Chain Of Thought)"},
    )
    cot_easy_eng: Optional[str] = field(
        default = 'As a smart student answer the given question.\
                    Read paragraph, and select only one answer between choices.',
        metadata = {'help' : "COT(Chain Of Thought) ENG ver"},
    )

@dataclass
class user_prompts:
    baseline: Optional[str] = field(
        default =
                """ 
                지문 :{paragraph}
                질문 :{question}
                선택지 :{choices}
                1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
                정답 :
                """,
        metadata = {'help' : "baseline"},
    )
    baseline_plus: Optional[str] = field(
        default=
                """
                지문 :{paragraph}
                질문 :{question}
                <보기> :{question_plus}
                선택지 :{choices}
                1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
                정답 :
                """
    )

    cot: Optional[str] = field(
        default =
                """
                다음 지문을 읽고, 주어진 질문에 답하세요.
                각 단계를 차례대로 설명하면서 문제를 풀이하세요. 
                지문: [지문 내용] 질문: [질문 내용] 
                단계 1: 지문에서 중요한 정보를 찾고, 질문에 필요한 핵심 요소를 파악하세요.
                단계 2: 지문에 언급된 내용과 질문을 비교하고 관련 정보를 추출하세요. 
                단계 3: 답을 도출하기 위해 필요한 근거를 제시하고, 최종적으로 답을 작성하세요.
                지문 :{paragraph}
                질문 :{question}
                선택지 :{choices}
                최종 정답 :
                """,
        metadata = {'help' : 'Planing'}
    )
    cot_plus: Optional[str] = field(
        default =
                """
                다음 지문을 읽고, 주어진 질문에 답하세요.
                각 단계를 차례대로 설명하면서 문제를 풀이하세요. 
                지문: [지문 내용] 질문: [질문 내용] 
                단계 1: 지문에서 중요한 정보를 찾고, 질문에 필요한 핵심 요소를 파악하세요.
                단계 2: 지문에 언급된 내용과 질문을 비교하고 관련 정보를 추출하세요. 
                단계 3: 답을 도출하기 위해 필요한 근거를 제시하고, 최종적으로 답을 작성하세요.
                지문 :{paragraph}
                질문 :{question}
                <보기> :{question_plus}
                선택지 :{choices}
                최종 정답 :
                """,
        metadata = {'help' : 'Planing'}
    )
    cot_eng: Optional[str] = field(
        default =  
                """
                As a smart student answer the given question.
                Read paragraph, and select only one answer between 5 choices.

                Paragraph :{paragraph}
                Question :{question}
                Choices :{choices}

                Choice one in 5 choices.
                Let's think step by step.
                answer :
                """,
        metadata = {'help' : "Planing"},
    )
    cot_eng_plus: Optional[str] = field(
        default =  
                """
                As a smart student answer the given question.
                Read paragraph, and select only one answer between 5 choices.

                Paragraph :{paragraph}
                Question :{question}
                More info :{question_plus}
                Choices :{choices}

                Choice one in 5 choices.
                Let's think step by step.
                answer :
                """,
        metadata = {'help' : "Planing"},
    )
    manual: Optional[str] = field(
        default =
                """
                국어 시험 문제를 푸는 대한민국의 고3 수험생으로서 위의 요약을 바탕으로 다음 문제의 답을 구하세요.\n
                문제를 풀이할 때, 반드시 지문을 참고하세요.\n
                문제는 무조건 1개의 정답만 있습니다.\n
                문제를 풀이할 때 모든 선택지들을 검토하세요.\n
                모든 선택지마다 정답인 근거와 정답이 아닌 근거를 설명하세요.\n
                다음의 형식을 따라 답변하세요.\n
                1번: (선택지 1번에 대한 답변) + "근거"\n
                2번: (선택지 2번에 대한 답변) + "근거"\n
                3번: (선택지 3번에 대한 답변) + "근거"\n
                4번: (선택지 4번에 대한 답변) + "근거"\n
                5번: (선택지 5번에 대한 답변) + "근거"\n
                최종 정답 :정답 번호\n
                지문 :{paragraph}
                질문 :{question}
                선택지 :{choices}
                """,
        metadata = {'help' : "메뉴얼을 상세히 지정"},)
    
    manual_plus: Optional[str] = field(
        default =
                """
                국어 시험 문제를 푸는 대한민국의 고3 수험생으로서 위의 요약을 바탕으로 다음 문제의 답을 구하세요.\n
                문제를 풀이할 때, 반드시 지문을 참고하세요.\n
                문제는 무조건 1개의 정답만 있습니다.\n
                문제를 풀이할 때 모든 선택지들을 검토하세요.\n
                모든 선택지마다 정답인 근거와 정답이 아닌 근거를 설명하세요.\n
                다음의 형식을 따라 답변하세요.\n
                1번: (선택지 1번에 대한 답변) + "근거"\n
                2번: (선택지 2번에 대한 답변) + "근거"\n
                3번: (선택지 3번에 대한 답변) + "근거"\n
                4번: (선택지 4번에 대한 답변) + "근거"\n
                5번: (선택지 5번에 대한 답변) + "근거"\n
                최종 정답 :정답 번호\n
                지문 :{paragraph}
                질문 :{question}
                <보기> :{question_plus}
                선택지 :{choices}
                """,
        metadata = {'help' : "메뉴얼을 상세히 지정"},)


    klue: Optional[str] = field(
        default =
                """
                지문 : {paragraph}
                질문 : {question}
                선택지 : {choices}
                지문을 읽고 지문에 답하세요.
                선택지 중 정답을 하나만 고르고 그 근거를 작성하세요.
                출력 형식 : 근거 : [고른 정답에 대한 근거] # [정답 번호]
                """,
        metadata = {'help' : "메뉴얼을 상세히 지정"},)
    
    klue_plus: Optional[str] = field(
        default =
                """
                지문 : {paragraph}
                질문 : {question}
                보기 : {question_plus}
                선택지 : {choices}
                지문을 읽고 지문에 답하세요.
                선택지 중 정답을 하나만 고르고 그 근거를 작성하세요.
                출력 형식 : 근거 : [고른 정답에 대한 근거] # [정답 번호]
                """,
        metadata = {'help' : "메뉴얼을 상세히 지정"},)


    klue_hint: Optional[str] = field(
        default =
                """
                지문 : {paragraph}
                질문 : {question}
                선택지 : {choices}
                {hint}
                지문을 읽고 지문에 답하세요.
                선택지 중 정답을 하나만 고르고 그 근거를 작성하세요.
                정답을 찾기 어렵다면 참고 문서를 참고하세요. 
                출력 형식 : 근거 : [고른 정답에 대한 근거] # [정답 번호]
                """,
        metadata = {'help' : "메뉴얼을 상세히 지정"},)
    
    klue_hint_plus: Optional[str] = field(
        default =
                """
                지문 : {paragraph}
                질문 : {question}
                보기 : {question_plus}
                선택지 : {choices}
                {hint}
                지문을 읽고 지문에 답하세요.
                선택지 중 정답을 하나만 고르고 그 근거를 작성하세요.
                정답을 찾기 어렵다면 참고 문서를 참고하세요.
                출력 형식 : 근거 : [고른 정답에 대한 근거] # [정답 번호]
                """,
        metadata = {'help' : "메뉴얼을 상세히 지정"},)

