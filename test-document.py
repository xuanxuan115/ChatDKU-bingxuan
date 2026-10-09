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

# 建库：为每段资料生成一个向量
documents = [
    {
        "source": "宿舍申请说明",
        "text": "学生申请宿舍，需要登录学生事务系统，填写住宿申请表并提交。",
    },
    {
        "source": "图书借阅说明",
        "text": "学生可以使用校园卡借阅图书，每次借阅期限为30天。",
    },
    {
        "source": "校园网络说明",
        "text": "连接校园无线网络后，需要使用学校账号和密码登录认证。",
    },
]
document_vectors = np.array(
    [embed_text(doc["text"]) for doc in documents],
    dtype=np.float32,
)

print("知识库向量矩阵形状：", document_vectors.shape)


def retrieve(question, top_k=2):
    # 问题也要用同一个 embedding 模型进行向量化
    query_vector = np.array(
        embed_text(question),
        dtype=np.float32,
    )

    # 余弦相似度：比较问题与每段资料的向量方向
    numerator = document_vectors @ query_vector
    denominator = (
        np.linalg.norm(document_vectors, axis=1)
        * np.linalg.norm(query_vector)
    )

    scores = numerator / np.maximum(denominator, 1e-12)

    # 从高到低排列，取最相关的 top_k 段
    indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            **documents[int(i)],
            "score": float(scores[i]),
        }
        for i in indices
    ]


question = "我想住在学校，应该怎么申请？"
results = retrieve(question)

for item in results:
    print(f"\n来源：{item['source']}")
    print(f"相似度：{item['score']:.4f}")
    print(f"内容：{item['text']}")