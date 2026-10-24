from openai import OpenAI

client = OpenAI(
    api_key="sk-e18a7b90337d4e059677cfd2df9f4175",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个简洁的技术助手"},
        {"role": "user", "content": "用一句话解释什么是向量数据库"}
    ],
)

print(response.choices[0].message.content)