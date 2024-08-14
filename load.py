from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 正确初始化jina-embeddings-v2-base-zh嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="jinaai/jina-embeddings-v2-base-zh",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 加载和处理文档
loader = TextLoader("/Users/liuxin/Downloads/aaa.txt", encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 创建向量存储
vector_store = FAISS.from_documents(texts, embeddings)

# 创建检索器
mmr_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 10})