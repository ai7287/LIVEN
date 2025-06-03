import streamlit as st
import os
import io
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from uuid import uuid4

from pydub import AudioSegment
from google.cloud import speech

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# ✅ 환경 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_key.json"
AudioSegment.converter = "C:/ffmpeg-7.1.1-essentials_build/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe"

# ✅ Streamlit 페이지 설정
st.set_page_config(page_title="세종대왕에게 묻다", page_icon="📜")
st.title("📜 세종대왕에게 묻다")
st.markdown("궁금한 것을 세종대왕께 여쭈어보세요. '짐이~ 하였도다'의 말투로 응답하십니다.")

# ✅ 세션 상태 초기화
if "chat_history" not in st.session_state:
    from langchain_community.chat_message_histories import StreamlitChatMessageHistory
    st.session_state["chat_history"] = StreamlitChatMessageHistory()

if "chat_log" not in st.session_state:
    st.session_state["chat_log"] = []

if "session_id" not in st.session_state:
    from uuid import uuid4
    st.session_state["session_id"] = str(uuid4())

# ✅ 이전 대화 기록 보여주기
for msg in st.session_state.get("chat_log", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✅ 프롬프트 템플릿
system_prompt = """
너는 조선의 제4대 임금 세종대왕이니라.
백성의 물음에 답할 때는 위엄 있고 따뜻한 말투로, "짐이", "~하였도다", "~하였느니라" 등을 사용하여 응답하라.
단, 한자는 사용하지 말고 순우리말과 한글로만 구성하라.
응답은 간결하되, 말의 끝에는 "그에 대해 더 물을 것이 있는가?" 혹은 이에 준하는 자연스러운 후속 질문으로 대화를 유도하라.
"""

# ✅ 체인 로딩 함수
@st.cache_resource
def load_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

    if not os.path.exists("faiss_index/index.faiss"):
        st.error("❌ FAISS 인덱스가 없습니다.")
        st.stop()

    db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1024,
        api_key=st.secrets["GROQ_API_KEY"]
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = StreamlitChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=st.session_state["chat_history"],
        output_key="answer"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{context}\n\n질문: {question}")
    ])

    base_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_type="similarity", k=5),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt, "document_variable_name": "context"},
        return_source_documents=True,
        output_key="answer"
    )

    return RunnableWithMessageHistory(
        base_chain,
        lambda _: st.session_state["chat_history"],
        input_messages_key="question",
        history_messages_key="chat_history"
    )

# ✅ 체인 초기화 (※ 꼭 버튼들보다 위에 있어야 함!)
chain = load_chain()

# ✅ 텍스트 입력
user_input = st.chat_input("💬 질문을 입력하세요")

# ✅ 🎤 음성 입력 버튼
if st.button("🎤 음성으로 질문하기"):
    fs = 48000
    duration = 5

    st.info("🎙️ 지금 말씀하세요... (5초 녹음 중)")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()

    # WAV 저장
    wav_buffer = io.BytesIO()
    wav.write(wav_buffer, fs, recording)
    wav_bytes = wav_buffer.getvalue()

    # Google STT
    try:
        client = speech.SpeechClient()
        audio_google = speech.RecognitionAudio(content=wav_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=fs,
            language_code="ko-KR"
        )
        response = client.recognize(config=config, audio=audio_google)

        if response.results:
            recognized_text = response.results[0].alternatives[0].transcript
            st.markdown(f"🗨️ 인식된 질문: **{recognized_text}**")

            if "session_id" not in st.session_state:
                st.session_state["session_id"] = str(uuid4())
            session_id = st.session_state["session_id"]

            with st.chat_message("user"):
                st.markdown(recognized_text)

            response = chain.invoke(
                {"question": recognized_text},
                config={"configurable": {"session_id": session_id}}
            )

            with st.chat_message("assistant"):
                st.markdown(response["answer"])

            # 로그 저장
            if "chat_log" not in st.session_state:
                st.session_state["chat_log"] = []
            st.session_state["chat_log"] += [
                {"role": "user", "content": recognized_text},
                {"role": "assistant", "content": response["answer"]}
            ]
        else:
            st.warning("❗ 음성을 인식하지 못했습니다.")

    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")

# ✅ 텍스트 입력 처리
if user_input:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid4())
    session_id = st.session_state["session_id"]

    with st.chat_message("user"):
        st.markdown(user_input)

    response = chain.invoke(
        {"question": user_input},
        config={"configurable": {"session_id": session_id}}
    )

    with st.chat_message("assistant"):
        st.markdown(response["answer"])

    if "chat_log" not in st.session_state:
        st.session_state["chat_log"] = []
    st.session_state["chat_log"] += [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response["answer"]}
    ]
