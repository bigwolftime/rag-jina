from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from load import mmr_retriever


# 创建RetrievalQA链
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=mmr_retriever,
    return_source_documents=True
)

# 使用函数
def answer_question(question):
    # 使用 MMR 检索器直接检索文档和分数
    docs_and_scores = mmr_retriever.get_relevant_documents(question, return_scores=True)
    # 分离文档和分数
    if isinstance(docs_and_scores[0], tuple):
        docs = [doc for doc, _ in docs_and_scores]
        scores = [score for _, score in docs_and_scores]
    else:
        docs = docs_and_scores
        # 如果没有返回分数，使用默认值
        scores = [1.0] * len(docs)
    # 使用 QA 链获取答案
    result = qa_chain({"query": question})
    return result["result"], docs, scores

def split(text):
    if len(text) <= 20:
        return text
    first_10 = text[:10]
    last_10 = text[-10:]
    return f"{first_10}...{last_10}"

def ask_and_print(question):
    answer, sources, scores = answer_question(question)
    print(f"问题: {question}")
    print(f"答案: {answer}")
    print("\n来源文档:")
    for i, (doc, score) in enumerate(zip(sources, scores)):
        print(f"文档 {i+1}:")
        split_content = split(doc.page_content)
        print(f"内容: {split_content}")
        print(f"相关性分数: {score}")
        print("-" * 50)