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
    weather = {"北京": "晴 18°C", "上海": "多云 22°C", "天津": "阴 17°C"}
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

messages = [{"role": "user", "content": "天津今天天气怎么样？"}]

response = client.chat.completions.create(
    model="glm-4.5-flash",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

msg = response.choices[0].message
print("第一轮模型返回:", msg)

if msg.tool_calls:
    messages.append(msg)
    for tool_call in msg.tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = get_weather(**args)
        print(f"本地执行: {tool_call.function.name}({args}) -> {result}")
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })

    final = client.chat.completions.create(
        model="glm-4.5-flash",
        messages=messages,
        tools=tools
    )
    print("最终回答:", final.choices[0].message.content)