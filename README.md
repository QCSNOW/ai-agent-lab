# AI Agent Lab

基于智谱 GLM / DeepSeek 大模型 API 的个人 AI Agent 实验项目，涵盖 RAG 检索增强问答与 Function Calling 工具调用。

## 项目结构

| 文件 | 说明 |
|------|------|
| `app.py` | Streamlit Web 版 RAG 知识库助手：上传文档 → Chroma 向量库 → 聊天问答 |
| `rag_demo.py` | 命令行版 RAG 原理演示：文档切块 → Embedding → 余弦相似度检索 → 大模型生成 |
| `agent_v3.py` | 多工具 Agent：天气查询、数学计算、时间查询、笔记检索（Function Calling） |

## 技术栈

- Python 3.12
- 智谱 GLM-4.5-Flash（OpenAI SDK 兼容接口）
- ChromaDB 向量数据库
- Streamlit Web UI
- NumPy 余弦相似度检索

## 快速开始

```bash
# 安装依赖
pip install openai python-dotenv chromadb streamlit numpy requests

# 配置 API Key
cp .env.example .env
# 编辑 .env 填入 ZHIPU_API_KEY

# 启动 Web 版 RAG 助手
streamlit run app.py

# 或运行命令行版 RAG 演示
python rag_demo.py

# 或运行多工具 Agent
python agent_v3.py
```

## 环境变量

在 `.env` 文件中配置：

```
ZHIPU_API_KEY=your_api_key_here
```
