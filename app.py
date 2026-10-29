import os
import chromadb
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# Chroma 向量数据库，持久化到本地文件夹
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("notes")

def embed(texts):
    resp = client.embeddings.create(model="embedding-3", input=texts)
    return [d.embedding for d in resp.data]

st.title("📚 个人知识库助手")

# 侧边栏：上传文档
with st.sidebar:
    st.header("上传文档")
    uploaded = st.file_uploader("选择 .txt 或 .md 文件", type=["txt", "md"])
    if uploaded and st.button("加入知识库"):
        text = uploaded.read().decode("utf-8")
        chunks = [c.strip() for c in text.split("\n") if c.strip()]
        vectors = embed(chunks)
        existing = collection.get()["ids"]
        added = 0
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            doc_id = f"{uploaded.name}_{i}"
            if doc_id not in existing:
                collection.add(documents=[chunk], embeddings=[vec], ids=[doc_id])
                added += 1
        st.success(f"已加入 {added} 个段落")

# 聊天界面
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("问点什么...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    qv = embed([query])
    results = collection.query(query_embeddings=qv, n_results=3)
    context = "\n".join(results["documents"][0]) if results["documents"] else ""

    with st.chat_message("assistant"):
        if context:
            with st.expander("查看检索到的文档"):
                st.write(context)
            messages = [
                {"role": "system", "content": f"根据以下文档回答，文档里没有就说不知道。\n\n文档：\n{context}"},
                {"role": "user", "content": query}
            ]
        else:
            messages = [{"role": "user", "content": query}]
        resp = client.chat.completions.create(model="glm-4.5-flash", messages=messages)
        answer = resp.choices[0].message.content
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})