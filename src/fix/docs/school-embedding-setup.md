# 使用学校 Qwen3-Embedding-8B 接入正式 ChatDKU

## 核对结论

本次上传的 config.py 已支持 provider=openai，setup.py 已调用 OpenAICompatibleEmbedding，上传的适配器已有同步、异步和批处理实现。这与此前仅含 local/tei 的代码版本不同。截图中 local 分支被调用，直接原因是运行时 provider 为 local；不是学校模型不被允许。

根目录 pyproject.toml 使用 src/chatdku。三个 Python 文件必须放入正式的 src/chatdku，不要只放在 src/chatdku-gpt/src/chatdku 副本中。

## 修改范围

- src/chatdku/config.py：auto 模式在有 OpenAI 模型和地址时选择 openai；错误提示涵盖三类接入。显式 local/tei/openai 始终优先。auto 同时具有多种配置时，新增的 OpenAI 完整配置优先于 TEI 和 local；推荐显式选择避免歧义。
- pyproject.toml：新增代码直接导入的 openai、httpx、pydantic 依赖声明。
- src/chatdku/setup.py：与上传文件相同，打包是为了让它被放到正式路径。
- src/chatdku/openai_embeddings.py：与上传文件相同，复用已有适配。
- school-embedding.env.example：配置示例，仅手动合并进根目录 .env，不覆盖其他配置。
- tests/test_embedding_config.py：8 项配置回归检查，无真实服务请求。

没有修改 loader、入库 pipeline、Qdrant、Agent 或源资料，也没有重新保存用户 FAQ。此包只是补丁，不是完整项目。

## 1. 放置文件

以 E:\dku的好多文件\student worker\ChatDKU\ChatDKU-Public-main 为根目录。
备份原文件，把本包 src/chatdku 中的三个文件覆盖到根目录相同位置；pyproject.toml 放到根目录，名称必须是 pyproject.toml，不能是 pyproject(1).toml。
其余 docs、tests 可按目录合并；不要删除原有文件夹。

## 2. 修改 .env

将 school-embedding.env.example 中的配置合并到根目录 .env。替换 Key 占位符；其他配置名称和模型名按示例。保留已有聊天模型配置。不要把真实 Key 发给别人或提交到 git。
openai 是接口格式，不表示使用 OpenAI 的模型或 Key。
先用 batch_size=1，减少学校网关批量输入大小带来的不确定性；确认支持批量后可提高。请求仍然是单元素字符串数组，符合兼容接口格式；若网关报 input 类型错误，需要依据学校接口规范调整适配器。
query instruction 先留空，与此前成功测试一致。如以后启用，需根据学校模型说明设置，并重新验证检索效果。

## 3. 在 CMD 中检查（截图中的终端是 CMD）

```bat
cd /d "E:\dku的好多文件\student worker\ChatDKU\ChatDKU-Public-main"
python -m uv sync
set "CHATDKU_EMBEDDING_PROVIDER=openai"
set "NO_PROXY=localhost,127.0.0.1,aichat.dukekunshan.edu.cn"
python -m uv run python -c "import chatdku.config as c; s=c.get_settings(); print(c.__file__); print('provider:',s.resolved_embedding_provider); print('model:',s.openai_embedding_model); print('url:',s.embedding_base_url)"
python -m uv run agent --check-embeddings
```

配置检查不打印 Key。文件路径应指向 src/chatdku/config.py，provider 应为 openai。
环境变量优先于 .env；若模型或 URL 输出不符合 .env，清除当前 CMD 同名旧变量后重新运行，例如 `set "CHATDKU_EMBEDDING_BASE_URL="`。清除终端变量后 .env 才能补入对应值。

预期模型构造日志为：

```text
Using OpenAI-compatible embedding model: Qwen3-Embedding-8B
Embedding health check passed; vector dimension: 4096
```

4096 是此前学校接口的实测维度，实际以当前接口返回为准。若仍出现 Using local embedding model，先核对运行目录、模块路径和 provider，不要继续下载本地模型。

PowerShell 设置变量语法不同：

```powershell
$env:CHATDKU_EMBEDDING_PROVIDER = "openai"
$env:NO_PROXY = "localhost,127.0.0.1,aichat.dukekunshan.edu.cn"
```

## 4. 正式入库和问答

把授权使用的一份 FAQ 源文件放在 data/documents/advising；不要重复放入同内容的 docx/md/json。依据当前 ingestion.md，原 Word 和整理后的 Markdown 都是支持的扩展名；自定义问答 JSON 是否按记录加载，需要另行核对 loader，此补丁没有改变该行为。

```bat
python -m uv run chatdku-ingest data/documents/advising
```

默认输出 data/index/qdrant 与 data/index/manifest.json。已有集合会被保护。只有确实要替换集合全部内容时才运行：

```bat
python -m uv run chatdku-ingest --replace data/documents/advising
```

更换 embedding 模型必须重建旧索引。独立脚本生成的 faq_vectors.npy 不会被此流程使用。

聊天模型需另配根目录 .env：

```dotenv
CHATDKU_LLM_MODEL=学校实际提供的聊天模型名称
CHATDKU_LLM_BASE_URL=https://aichat.dukekunshan.edu.cn:3443/v1
CHATDKU_LLM_API_KEY=学校APIKey
```

上面的聊天 URL 仅适用于学校聊天服务使用同一个地址。使用实际聊天模型标识，不填 embedding 模型，不猜模型名称。

```bat
python -m uv run agent --no-rewrite "How often should I meet with my advisor?"
python -m uv run agent
```

## 验证范围

已通过 Python 语法编译、TOML 解析和 8 项配置逻辑测试。当前执行环境缺少 dotenv/OpenAI/HTTPX/LlamaIndex；运行配置逻辑测试时，用无操作替身替换 dotenv 文件加载，其余配置代码实际执行。测试未验证 .env 文件解析、真实依赖兼容、学校联网、Qdrant 入库或 Agent 端到端流程；这些需要在你的项目环境运行上述命令确认。

安装依赖后可在项目根目录自行运行原始测试（使用真实 dotenv 导入；测试用 mock 隔离文件和环境）：

```bat
python -m uv run python -m unittest discover -s tests -p test_embedding_config.py -v
```

适配器保持 `client.embeddings.create(model=..., input=..., encoding_format="float")` 调用；接口形式参考官方文档：https://developers.openai.com/api/reference/python/resources/embeddings/methods/create 。学校权限与参数支持以实际返回为准。
