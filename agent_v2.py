import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import requests

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ===== 工具本体：普通 Python 函数 =====

def get_weather(city: str) -> str:
    # 第一步：城市名 -> 经纬度
    geo = requests.get(
        f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=zh"
    ).json()
    if not geo.get("results"):
        return f"找不到城市：{city}"
    loc = geo["results"][0]
    lat, lon, name = loc["latitude"], loc["longitude"], loc["name"]

    # 第二步：经纬度 -> 实时天气
    resp = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code"
    ).json()
    temp = resp["current"]["temperature_2m"]
    code = resp["current"]["weather_code"]

    weather_map = {0:"晴", 1:"大部晴", 2:"多云", 3:"阴", 45:"雾", 51:"小毛毛雨", 61:"小雨", 63:"中雨", 80:"阵雨", 95:"雷阵雨"}
    desc = weather_map.get(code, f"天气代码{code}")
    return f"{name}：{desc}，{temp}°C"


def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")

# ===== 给模型看的工具说明书 =====

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，支持加减乘除括号，如 1+2*3、(10+5)/3",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "数学表达式"}},
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

# ===== 路由表：工具名 -> 实际函数 =====

tool_map = {
    "get_weather": get_weather,
    "calculate": calculate,
    "get_current_time": get_current_time
}

# ===== 主循环 =====

messages = [{"role": "system", "content": "你是一个有用的助手。需要查天气、算数学题或查时间时调用对应工具。"}]

while True:
    user_input = input("\n你：")
    if user_input == "exit":
        break
    messages.append({"role": "user", "content": user_input})

    while True:
        response = client.chat.completions.create(
            model="glm-4.5-flash",
            messages=messages,
            tools=tools,
            tool_choice="auto"
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
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })