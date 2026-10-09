from getpass import getpass
from openai import OpenAI

# 1. 创建连接客户端
# 如果之前成功的配置还有其他参数，请一并保留
client = OpenAI(
    base_url="https://aichat.dukekunshan.edu.cn:3443/v1",
    api_key=getpass("请输入学校 API Key输入时不显示"),
)

# 2. 指定模型名称
EMBEDDING_MODEL = "Qwen3-Embedding-8B"

# 3. 定义向量化函数
def embed_text(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        encoding_format="float",
    )
    return response.data[0].embedding

# 4. 输入文本并调用函数
text = "我想了解如何申请学校宿舍。"
vector = embed_text(text)

print("向量维度:", len(vector))
print("前5个数字:", vector[:5])