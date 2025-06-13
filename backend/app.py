import streamlit as st
import os
import io
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from uuid import uuid4
import sounddevice as sd
import scipy.io.wavfile as wav
from pydub import AudioSegment
from google.cloud import speech
from gtts import gTTS
import tempfile

from transformers import AutoTokenizer, AutoModel
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def play_tts(text, lang='ko'):
    tts = gTTS(text=text, lang=lang)

    # 임시 파일 생성 (삭제 지연)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_path = fp.name

    # gTTS로 음성 저장
    tts.save(temp_path)

    # 저장된 파일 열어서 Streamlit으로 재생
    with open(temp_path, "rb") as f:
        audio_bytes = f.read()
        st.audio(io.BytesIO(audio_bytes), format='audio/mp3')

    # 파일 수동 삭제
    os.remove(temp_path)



# ✅ 감정 분석기 모델 정의 및 로딩
MODEL_NAME = "skt/kobert-base-v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
df = pd.read_csv("finish_emotionpreprocess.csv", encoding='utf-8').dropna()
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['emotion'])

class KoBERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=7, dropout_rate=0.1):
        super(KoBERTClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        out = self.dropout(pooled_output)
        return self.classifier(out)

bert_model = AutoModel.from_pretrained(MODEL_NAME)
emo_model = KoBERTClassifier(bert_model, num_classes=len(label_encoder.classes_)).to(device)
emo_model.load_state_dict(torch.load("kobert_emotion_model3.pth", map_location=device), strict=False)
emo_model.eval()

def predict_emotion(text):
    with torch.no_grad():
        encoded = tokenizer(text, truncation=True, padding='max_length', max_length=64, return_tensors='pt')
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        output = emo_model(input_ids, attention_mask)
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]
        return label_encoder.inverse_transform([pred])[0]

# ✅ Streamlit 페이지 설정
st.set_page_config(page_title="세종대왕에게 묻다", page_icon="📜")
st.title("📜 세종대왕에게 묻다")
st.markdown("세종대왕과의 대화 및 감정 분석 서비스를 제공합니다.")

# ✅ 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = StreamlitChatMessageHistory()
if "chat_log" not in st.session_state:
    st.session_state["chat_log"] = []
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid4())

for msg in st.session_state["chat_log"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✅ LangChain 체인 로딩
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

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=st.session_state["chat_history"],
        output_key="answer"
    )

    prompt = ChatPromptTemplate.from_messages([
    ("system", """
너는 조선의 제4대 임금 세종대왕이니라.

너는 반드시 아래 지시를 철저히 따르라:

1. 오직 세종대왕의 말투로만 응답하라.  
   예: "짐이", "~하였도다", "~하였느니라", "~이니라", "~노라" 등의 말투를 사용할 것.

2. 절대로 현대식 존댓말(예: ~입니다, ~하세요, ~하시나요)을 사용하지 말라.  
   말투는 엄중하고 품격 있게 유지하되, 따뜻함을 잃지 말라.

3. 사용자가 반말 또는 예의 없는 말투(예: “뭐 했어?”, “밥 먹었냐?”, “응?”)로 질문할 경우,  
   세종대왕으로서의 위엄을 지키되 **너그럽고 인자한 말투로 공손함을 권유하라.**  
   지나치게 꾸짖지 말고, **자비로움과 교훈을 담아 말하라.**  
   사용자가 무안을 느끼지 않도록 하며, 다음부터 예를 갖추어 말해 주기를 바라는 식으로 표현하라.

4. 한자나 외래어는 절대 사용하지 말라.  
   예: '汝', '學', '制度', '의학', '교육', 'IT', 'AI' 등의 글자는 모두 금지이다.  
   반드시 순우리말과 한글로만 응답하라.

5. '내 생각에는', '고려해 보건대', '<think>...</think>'와 같은 사고 흐름이나 설명은 **절대로 출력하지 말라.**  
   오직 세종대왕의 응답 문장만 출력하라.

6. 대답은 간결하고 위엄 있게 하되, 말 끝에는 항상 "그에 대해 더 물을 것이 있는가?" 또는 이에 준하는 후속 질문을 붙일 것.

7. 질문자의 말투나 질문 배경을 유추하거나 해석하지 말고, **질문 내용 자체에만 충실히 응답하라.**
"""),
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

chain = load_chain()

# ✅ 텍스트 입력 처리
user_input = st.chat_input("💬 질문을 입력하세요")
if user_input:
    session_id = st.session_state["session_id"]
    with st.chat_message("user"):
        st.markdown(user_input)

    response = chain.invoke({"question": user_input}, config={"configurable": {"session_id": session_id}})

    with st.chat_message("assistant"):
        st.markdown(response["answer"])
        emotion = predict_emotion(response["answer"])
        st.caption(f"✳️ 감정 배열 결과: {emotion}")
        play_tts(response["answer"])

    st.session_state["chat_log"] += [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response["answer"]}
    ]

# ✅ 🎤 음성 질문 처리
if st.button("🎤 음성으로 질문하기"):
    fs = 48000
    duration = 5
    st.info("🎤 지금 말씀하세요... (5초 녹음 중)")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav_buffer = io.BytesIO()
    wav.write(wav_buffer, fs, recording)
    wav_bytes = wav_buffer.getvalue()

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
            st.markdown(f"🔈인심된 질문: **{recognized_text}**")

            session_id = st.session_state["session_id"]
            with st.chat_message("user"):
                st.markdown(recognized_text)

            response = chain.invoke({"question": recognized_text}, config={"configurable": {"session_id": session_id}})

            with st.chat_message("assistant"):
                st.markdown(response["answer"])
                emotion = predict_emotion(response["answer"])
                st.caption(f"✳️ 감정 배열 결과: {emotion}")
                play_tts(response["answer"])

            st.session_state["chat_log"] += [
                {"role": "user", "content": recognized_text},
                {"role": "assistant", "content": response["answer"]}
            ]
        else:
            st.warning("❗ 음성을 인심하지 못했습니다.")

    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
