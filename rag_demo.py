import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ===== 1. 你的文档（以后换成真 PDF/txt 就行）=====

notes = """
智谱 AI Agent 学习笔记

1. 环境搭建：Python 3.12，虚拟环境 .venv，pip 清华镜像。
2. Git 工程化：add、commit、push，远程仓库 GitHub。
3. 大模型调用：用 OpenAI SDK 兼容智谱 GLM-4.5-Flash。
4. 多轮对话：messages 列表存历史，system/user/assistant 三种角色。
5. Function Calling：模型决定调哪个工具，Python 本地执行，结果回传。
6. Agent 循环：模型看一步、调一个工具、看结果、再决定下一步。
7. RAG 原理：文档切块、向量化、问题匹配相关块、塞进 prompt。
"""

# ===== 2. 切块 + 向量化 =====

chunks = [c.strip() for c in notes.strip().split("\n") if c.strip()]

def embed(texts):
    resp = client.embeddings.create(model="embedding-3", input=texts)
    return [d.embedding for d in resp.data]

chunk_vectors = embed(chunks)

# ===== 3. 余弦相似度检索 =====

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve(query, top_k=2):
    qv = embed([query])[0]
    scores = [cosine_sim(qv, cv) for cv in chunk_vectors]
    ranked = sorted(zip(scores, chunks), reverse=True)
    return [c for s, c in ranked[:top_k]]

# ===== 4. 问答 =====

while True:
    q = input("\n你：")
    if q == "exit":
        break

    relevant = retrieve(q)
    print("  [检索到相关段落]")
    for r in relevant:
        print(f"    - {r}")

    context = "\n".join(relevant)
    messages = [
        {"role": "system", "content": f"根据以下笔记回答问题，笔记里没有就说不知道。\n\n笔记：\n{context}"},
        {"role": "user", "content": q}
    ]
    resp = client.chat.completions.create(model="glm-4.5-flash", messages=messages)
    print("助手:", resp.choices[0].message.content)