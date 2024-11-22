import requests
from bs4 import BeautifulSoup
from konlpy.tag import Okt
import pandas as pd

# Okt 형태소 분석기 사용하여 명사 추출 함수
okt = Okt()

def extract_keywords(text):
    """ paragraph에서 명사 추출 """
    nouns = okt.nouns(text)
    return " ".join(nouns)

# 예시 데이터 (DataFrame 형태)
data = {
    'id': ['generation-for-nlp-470'],
    'paragraph': ['직지 심체요절은 백운화상이 저술한 책을 청주 흥덕사에서 1377년 7월에 금속활자로 인쇄한 것이다. 1972년 ‘세계도서의 해’에 출품되어 세계 최고의 금속활자본으로 공인되었다. 이 책은 이러한 가치를 인정받아 2001년 9월에 유네스코 세계기록유산으로 등재되었다.'],
    'problems': ["{'question': '다음 문화유산이 간행된 왕대에 대한 설명으로 옳은 것은?', 'choices': ['원황실은 북쪽으로 도망가고 명이 건국되었다.', '기존의 토지문서를 불태워버리고 과전법을 시행하였다.', '원에만 권당을 설치하여 고려와 원의 지식인들이 교류하였다.', '명은 철령위를 설치한다고 고려에 통보하였다.'], 'answer': 3}"],
    'question_plus': [None]  # NaN 처리 예시
}

# DataFrame 생성
df = pd.DataFrame(data)

# 'paragraph'에서 키워드 추출
df['keywords'] = df['paragraph'].apply(extract_keywords)

# Google Custom Search API로 인터넷 검색 함수
def google_search(query, api_key, cx):
    """ Google Custom Search API를 사용하여 검색 결과 얻기 """
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # JSON 형식으로 결과 반환
    else:
        return None

# 검색 결과 크롤링 함수 (상위 3개만 추출)
def crawl_results(results):
    """ 크롤링한 검색 결과에서 링크와 타이틀을 추출 (상위 3개) """
    crawled_data = []
    for item in results['items'][:3]:  # 상위 3개만 가져옴
        title = item['title']
        link = item['link']
        
        # 웹 페이지 크롤링
        page = requests.get(link)
        soup = BeautifulSoup(page.text, 'html.parser')
        content = soup.get_text()  # 페이지 전체 텍스트
        
        crawled_data.append({
            'title': title,
            'link': link,
            'content': content[:500]  # 내용의 처음 500자만 가져오기
        })
    
    # link 열 제거
    for data in crawled_data:
        del data['link']  # 'link' 열 제거
    
    return crawled_data

# Google Custom Search API 키와 검색 엔진 ID
API_KEY = 'YOUR_GOOGLE_API_KEY'
CX = 'YOUR_GOOGLE_CUSTOM_SEARCH_ENGINE_ID'

# 키워드를 추출하여 Google에서 검색
keywords = df['keywords'].iloc[0]  # 첫 번째 문서의 키워드를 사용
search_results = google_search(keywords, API_KEY, CX)

# 검색 결과가 존재하면 크롤링 시작
if search_results:
    crawled_data = crawl_results(search_results)
    
    # 크롤링된 데이터를 출력
    for result in crawled_data:
        print(f"Title: {result['title']}")
        print(f"Content (first 500 chars): {result['content']}")
        print("="*50)  # 구분선 추가
else:
    print("검색 결과가 없습니다.")



#-------------------------------#



import requests
from bs4 import BeautifulSoup
from konlpy.tag import Okt
import pandas as pd
from tqdm import tqdm

# Okt 형태소 분석기 사용하여 명사 추출 함수
okt = Okt()

def extract_keywords(text):
    """ paragraph에서 명사 추출 """
    nouns = okt.nouns(text)
    return " ".join(nouns)

# 예시 데이터 (DataFrame 형태)
data = {
    'id': ['generation-for-nlp-470'],
    'paragraph': ['직지 심체요절은 백운화상이 저술한 책을 청주 흥덕사에서 1377년 7월에 금속활자로 인쇄한 것이다. 1972년 ‘세계도서의 해’에 출품되어 세계 최고의 금속활자본으로 공인되었다. 이 책은 이러한 가치를 인정받아 2001년 9월에 유네스코 세계기록유산으로 등재되었다.'],
    'problems': ["{'question': '다음 문화유산이 간행된 왕대에 대한 설명으로 옳은 것은?', 'choices': ['원황실은 북쪽으로 도망가고 명이 건국되었다.', '기존의 토지문서를 불태워버리고 과전법을 시행하였다.', '원에만 권당을 설치하여 고려와 원의 지식인들이 교류하였다.', '명은 철령위를 설치한다고 고려에 통보하였다.'], 'answer': 3}"],
    'question_plus': [None]  # NaN 처리 예시
}

# DataFrame 생성
df = pd.DataFrame(data)

# 'paragraph'에서 키워드 추출
df['keywords'] = df['paragraph'].apply(extract_keywords)

# Google Custom Search API로 인터넷 검색 함수
def google_search(query, api_key, cx):
    """ Google Custom Search API를 사용하여 검색 결과 얻기 """
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # JSON 형식으로 결과 반환
    else:
        return None

# 검색 결과 크롤링 함수 (상위 3개만 추출)
def crawl_results(results):
    """ 크롤링한 검색 결과에서 링크와 타이틀을 추출 (상위 3개) """
    crawled_data = []
    for item in results['items'][:3]:  # 상위 3개만 가져옴
        title = item['title']
        link = item['link']
        
        # 웹 페이지 크롤링
        page = requests.get(link)
        soup = BeautifulSoup(page.text, 'html.parser')
        content = soup.get_text()  # 페이지 전체 텍스트
        
        crawled_data.append({
            'title': title,
            'link': link,
            'content': content[:500]  # 내용의 처음 500자만 가져오기
        })
    
    return crawled_data

# Google Custom Search API 키와 검색 엔진 ID
API_KEY = 'YOUR_GOOGLE_API_KEY'
CX = 'YOUR_GOOGLE_CUSTOM_SEARCH_ENGINE_ID'

# 크롤링된 데이터를 힌트로 제공하는 함수
def provide_hint_with_crawled_data(paragraph, crawled_data=None):
    """ 크롤링된 데이터를 힌트로 제공 """
    hint = "다음은 관련된 웹 검색 결과에서 추출한 정보입니다.\n"

    if crawled_data:
        # 검색된 데이터에서 상위 3개 항목을 힌트로 사용
        for data in crawled_data:
            hint += f"\n[제목] {data['title']}\n[내용] {data['content']}\n"

    return hint

# 힌트 제공
hint_list = []
for i in tqdm(range(len(df))):
    paragraph = df.iloc[i]['paragraph']
    question = df.iloc[i]['problems']
    choices = eval(df.iloc[i]['problems'])['choices']  # 문자열로 되어 있는 'problems'을 딕셔너리로 변환
    question_plus = df.iloc[i].get('question_plus', '')  # 질문에 추가적인 정보가 있는지 확인

    # 키워드를 추출하여 Google에서 검색
    keywords = extract_keywords(paragraph)
    search_results = google_search(keywords, API_KEY, CX)

    if search_results:
        crawled_data = crawl_results(search_results)
    else:
        crawled_data = None

    # 크롤링된 데이터를 그대로 힌트로 사용
    hint = provide_hint_with_crawled_data(paragraph, crawled_data)

    # 생성된 힌트를 paragraph 앞에 붙이기
    updated_paragraph = hint + "\n" + paragraph
    df.at[i, 'paragraph'] = updated_paragraph  # 힌트를 paragraph 앞에 추가

    hint_list.append(hint)

# 이제 'df'에 수정된 paragraph가 반영됩니다.
