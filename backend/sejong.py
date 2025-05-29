import streamlit as st
import os
import pickle
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# 페이지 설정
st.set_page_config(page_title="세종대왕에게 묻다", page_icon="📜")
st.title("📜 세종대왕에게 묻다")
st.markdown("궁금한 것을 세종대왕께 여쭈어보세요. '짐이~ 하였도다'의 말투로 응답하십니다.")

for msg in st.session_state.get("chat_log", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력
user_input = st.chat_input("💬 질문을 입력하세요")

# 프롬프트 템플릿
system_prompt = """
너는 조선의 제4대 임금 세종대왕이니라.
백성의 물음에 답할 때는 위엄 있고 따뜻한 말투로, "짐이", "~하였도다", "~하였느니라" 등을 사용하여 응답하라.
단, 한자는 사용하지 말고 순우리말과 한글로만 구성하라.
응답은 간결하되, 말의 끝에는 "그에 대해 더 물을 것이 있는가?" 혹은 이에 준하는 자연스러운 후속 질문으로 대화를 유도하라.
"""

# 체인 로딩 함수
@st.cache_resource
def load_chain():
    # 임베딩 모델 로딩
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

    # FAISS 인덱스 존재 여부 확인
    if not os.path.exists("faiss_index/index.faiss"):
        st.error("❌ FAISS 인덱스 파일이 존재하지 않습니다. 먼저 인덱스를 생성해주세요.")
        st.stop()

    # FAISS 인덱스 로드
    db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

    # LLM 설정
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1024,
        api_key=st.secrets["GROQ_API_KEY"]
    )

    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_community.chat_message_histories import StreamlitChatMessageHistory
    from uuid import uuid4

    # 메시지 기록 및 메모리
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = StreamlitChatMessageHistory()
    chat_history = st.session_state["chat_history"]
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=chat_history,
        output_key="answer"
    )

    from langchain.prompts import ChatPromptTemplate

    # Prompt template for combining retrieved documents and the question
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{context}\n\n질문: {question}")
    ])

    from langchain.chains.combine_documents import create_stuff_documents_chain

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_variable_name="context"
    )

    base_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_type="similarity", k=5),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt, "document_variable_name": "context"},
        return_source_documents=True,
        output_key="answer"
    )

    chain = RunnableWithMessageHistory(
        base_chain,
        lambda _: chat_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )
    return chain

# 체인 초기화
chain = load_chain()

# 질문 처리
if user_input:
    from uuid import uuid4
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

    # Save chat log
    if "chat_log" not in st.session_state:
        st.session_state["chat_log"] = []
    st.session_state["chat_log"].append({"role": "user", "content": user_input})
    st.session_state["chat_log"].append({"role": "assistant", "content": response["answer"]})