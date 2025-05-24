import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# JSON QA 데이터 로딩
with open("qa_with_source.json", "r", encoding="utf-8") as f:
    qa_list = json.load(f)

# Document로 변환
docs = []
for item in qa_list:
    question = item.get("question", "")
    answer = item.get("answer", "")
    content = f"질문: {question}\n답변: {answer}"
    docs.append(Document(page_content=content, metadata={"number": item.get("number", -1)}))

# 임베딩 모델 로딩
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

# FAISS 인덱스 생성 및 저장
db = FAISS.from_documents(docs, embedding_model)
db.save_local("faiss_index")