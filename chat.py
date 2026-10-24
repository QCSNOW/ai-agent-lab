import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

messages = [
    {"role": "system", "content": "你是一个简洁的技术助手"}
]

while True:
    user_input = input("你: ")
    if user_input == "exit":
        break
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )
    answer = response.choices[0].message.content
    print("助手:", answer)
    messages.append({"role": "assistant", "content": answer})