import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage, ChatMessage
from pdfminer.high_level import extract_text

# 페이지 설정
st.set_page_config(page_title='학습용 영문 저널 번역/요약 봇', layout='wide')
st.title('📚 Journal Translator & Summarizer')

# 사이드바 설정
with st.sidebar:
    st.header("🔑 API 설정")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.header("📄 문서 업로드")
    uploaded_file = st.file_uploader("영문 저널(PDF)을 업로드하세요", type=['pdf'])

# RAG 프로세스 정의
def process_document(file):
    # 1. Loading
    text = extract_text(file)
    # 2. Splitting: 저널의 복잡한 문장 구조를 고려해 Recursive 분할 권장
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    # 3. Embedding & 4. Storage
    vs = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return vs, text

if uploaded_file and api_key:
    if 'vs' not in st.session_state:
        with st.spinner('문서를 분석하고 있습니다...'):
            vs, raw_text = process_document(uploaded_file)
            st.session_state.vs = vs
            st.session_state.raw_text = raw_text
        st.success('분석 완료!')

# 대화 로그 관리
if "messages" not in st.session_state:
    st.session_state.messages = [ChatMessage(role="assistant", content="학습용 저널을 업로드하면 한국어로 번역 및 요약을 도와드립니다!")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

# 5. Retrieval & Generation
if prompt := st.chat_input("질문을 입력하거나 '요약'이라고 입력하세요"):
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    if 'vs' in st.session_state:
        # 검색(Retrieval): 질문과 관련된 영문 단락 3개 추출
        docs = st.session_state.vs.similarity_search(prompt, k=3)
        context = "\n".join([d.page_content for d in docs])

        # 생성(Generation) 및 프롬프트 엔지니어링
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        
        # 한국어 번역 및 학습용 요약을 위한 시스템 프롬프트
        sys_msg = SystemMessage(content="""너는 영문 학술 저널을 한국어로 번역하고 요약해주는 학습 보조 전문가야.
        1. 제공된 영문 맥락(Context)을 바탕으로 사용자의 질문에 한국어로 답해줘.
        2. 전문 용어는 영어 원문을 괄호안에 병기해줘. (예: 인지 구조(Cognitive Architecture))
        3. 요약 요청 시 핵심 내용을 Notion 스타일의 불렛포인트로 정리해줘.
        4. 문서에 없는 내용은 추측하지 말고 '제공된 문서에서 찾을 수 없습니다'라고 답해줘.""")
        
        human_msg = HumanMessage(content=f"질문: {prompt}\n\n영문 맥락:\n{context}")

        with st.chat_message("assistant"):
            response = llm.invoke([sys_msg, human_msg])
            st.write(response.content)
            st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))