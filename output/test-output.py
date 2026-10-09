import json
from pathlib import Path
import numpy as np
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



# 获取当前脚本所在目录，即 output 文件夹
BASE_DIR = Path(__file__).resolve().parent

# 读取同目录中的 FAQ 数据
payload = json.loads(
    (BASE_DIR / "faq_records.json").read_text(encoding="utf-8")
)
records = payload["records"]

# 与前面 RAG 示例中的 documents 结构保持一致
documents = [
    {
        "source": item["source"],
        "text": item["material_text"],
        "embedding_text": item["embedding_text"],
    }
    for item in records
]

# 每组问答生成一个向量
document_vectors = np.array(
    [embed_text(doc["embedding_text"]) for doc in documents],
    dtype=np.float32,
)

print("问答数量：", len(documents))
print("向量矩阵形状：", document_vectors.shape)

# 保存，避免后续每次启动都重新调用模型
np.save(BASE_DIR / "faq_vectors.npy", document_vectors)