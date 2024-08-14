## Usage

### 1. Prepare environment

```shell

conda create --name rag-jina python=3.10

conda activate rag-jina

pip install -r requirements.txt

```

### 2. A simple example(optional)

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
import os

# 设置OpenAI API密钥
os.environ["OPENAI_API_KEY"] = "key"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 正确初始化jina-embeddings-v2-base-zh嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="jinaai/jina-embeddings-v2-base-zh",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 加载和处理文档
loader = TextLoader("./1.txt", encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 创建向量存储
vectorstore = FAISS.from_documents(texts, embeddings)

# 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 初始化语言模型（这里使用OpenAI，但您可以替换为其他支持中文的模型）
llm = OpenAI(temperature=0)

# 创建RetrievalQA链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 使用函数
def answer_question(question):
    result = qa_chain({"query": question})
    print(f"66666: {result}")
    return result["result"], result["source_documents"]

# 示例使用
question = "这本书写了什么？"
answer, sources = answer_question(question)

print(f"问题: {question}")
print(f"答案: {answer}")
print("\n来源文档:")
for i, doc in enumerate(sources):
    print(f"文档 {i+1}:")
    print(doc.page_content)
    print("-" * 50)
```