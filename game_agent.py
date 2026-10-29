import os
import json
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ===== 模拟游戏数据 =====

def get_daily_overview(days: int = 1) -> str:
    """获取近 days 天的游戏总览数据"""
    today = datetime.now()
    lines = []
    for i in range(days - 1, -1, -1):
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        # 模拟合理数据：基础值加波动
        downloads = 12000 + i * 300 + (i % 3) * -500
        revenue = downloads * 4.8
        dau = 35000 + i * 800
        lines.append(f"{d}：下载{downloads}，收入{int(revenue)}元，DAU{dau}")
    return "\n".join(lines)

def get_channel_detail(date: str) -> str:
    """获取某日分渠道明细"""
    data = {
        "抖音": {"downloads": 5200, "revenue": 26000},
        "微信": {"downloads": 4100, "revenue": 19500},
        "快手": {"downloads": 1800, "revenue": 8200},
        "B站": {"downloads": 900, "revenue": 4300},
    }
    lines = [f"{date} 分渠道："]
    for ch, v in data.items():
        lines.append(f"  {ch}：下载{v['downloads']}，收入{v['revenue']}元")
    return "\n".join(lines)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_daily_overview",
            "description": "获取游戏近几天的下载量、收入、DAU总览",
            "parameters": {
                "type": "object",
                "properties": {"days": {"type": "integer", "description": "查询最近几天"}},
                "required": ["days"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_channel_detail",
            "description": "获取某日分渠道下载和收入明细",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string", "description": "日期，如2026-10-29"}},
                "required": ["date"]
            }
        }
    }
]

tool_map = {"get_daily_overview": get_daily_overview, "get_channel_detail": get_channel_detail}

messages = [{"role": "system", "content": "你是游戏运营数据助手，帮运营同学分析游戏数据。需要数据时调用工具，用中文生成简洁的日报或分析。"}]

while True:
    user_input = input("\n你：")
    if user_input == "exit":
        break
    messages.append({"role": "user", "content": user_input})
    while True:
        resp = client.chat.completions.create(
            model="glm-4.5-flash", messages=messages, tools=tools, tool_choice="auto"
        )
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            print("助手:", msg.content)
            break
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            result = tool_map[tc.function.name](**args)
            print(f"  [调用] {tc.function.name}({args})")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})