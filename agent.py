import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

def get_weather(city: str) -> str:
    weather = {"北京": "晴 18°C", "上海": "多云 22°C", "天津": "阴 17°C", "广州": "雷阵雨 28°C"}
    return weather.get(city, f"{city}：暂无数据")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如北京、上海"}
                },
                "required": ["city"]
            }
        }
    }
]

messages = [{"role": "system", "content": "你是一个有用的助手，需要查天气时调用工具。"}]

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
            args = json.loads(tc.function.arguments)
            result = get_weather(**args)
            print(f"  [调用工具] {tc.function.name}({args}) -> {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })