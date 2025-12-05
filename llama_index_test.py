import os
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from customer_embedding import CustomerOllamaEmbedding
from llama_index.core import StorageContext, load_index_from_storage


Settings.llm = Ollama(
    model="qwen:7b",
    base_url="http://localhost:11434",
    temperature=0.1,
    context_window=8192,
    request_timeout=600
)
Settings.embed_model = CustomerOllamaEmbedding(
    model_name="bge-large",  # 用千问模型做嵌入（Ollama自动处理）
    base_url="http://localhost:11434",
    client_kwargs={"timeout": 600}
)

os.makedirs("./docs", exist_ok=True)
os.makedirs("./storage", exist_ok=True)

documents = SimpleDirectoryReader(input_files=["./docs/三体.txt"]).load_data()

# index = VectorStoreIndex.from_documents(documents, show_progress=True)
# index.storage_context.persist(persist_dir="./storage")

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(similarity_top_k=20, timeout=600)

response = query_engine.query("汪淼的故事线是怎样的？")
print(response.response)



