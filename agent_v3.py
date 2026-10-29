import os
import json
import requests
import numpy as np
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ===== 工具1：真天气 =====

def get_weather(city: str) -> str:
    geo = requests.get(
        f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=zh"
    ).json()
    if not geo.get("results"):
        return f"找不到城市：{city}"
    loc = geo["results"][0]
    resp = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={loc['latitude']}&longitude={loc['longitude']}&current=temperature_2m,weather_code"
    ).json()
    wmap = {0:"晴",1:"大部晴",2:"多云",3:"阴",45:"雾",61:"小雨",80:"阵雨",95:"雷阵雨"}
    code = resp["current"]["weather_code"]
    return f"{loc['name']}：{wmap.get(code, f'代码{code}')}，{resp['current']['temperature_2m']}°C"

# ===== 工具2：计算 =====

def calculate(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算错误: {e}"

# ===== 工具3：时间 =====

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")

# ===== 工具4：RAG 查笔记 =====

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

chunks = [c.strip() for c in notes.strip().split("\n") if c.strip()]

def embed(texts):
    resp = client.embeddings.create(model="embedding-3", input=texts)
    return [d.embedding for d in resp.data]

chunk_vectors = embed(chunks)

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def search_notes(query: str) -> str:
    qv = embed([query])[0]
    scores = [cosine_sim(qv, cv) for cv in chunk_vectors]
    ranked = sorted(zip(scores, chunks), reverse=True)
    return "\n".join([c for _, c in ranked[:2]])

# ===== 工具说明书 + 路由 =====

tools = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "查询城市实时天气",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }},
    {"type": "function", "function": {
        "name": "calculate", "description": "计算数学表达式",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
    }},
    {"type": "function", "function": {
        "name": "get_current_time", "description": "获取当前时间",
        "parameters": {"type": "object", "properties": {}}
    }},
    {"type": "function", "function": {
        "name": "search_notes", "description": "搜索用户自己的学习笔记，用户问笔记内容、学习记录、之前学过什么时调用",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    }}
]

tool_map = {
    "get_weather": get_weather,
    "calculate": calculate,
    "get_current_time": get_current_time,
    "search_notes": search_notes
}

# ===== 主循环 =====

messages = [{"role": "system", "content": "你是个人助手。查天气、算数、时间、学习笔记时调用对应工具。"}]

while True:
    user_input = input("\n你：")
    if user_input == "exit":
        break
    messages.append({"role": "user", "content": user_input})

    while True:
        response = client.chat.completions.create(
            model="glm-4.5-flash", messages=messages, tools=tools, tool_choice="auto"
        )
        msg = response.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            print("助手:", msg.content)
            break
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            result = tool_map[tc.function.name](**args)
            print(f"  [调用] {tc.function.name}({args}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})