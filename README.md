## Usage

### 1. Prepare environment

```shell

conda create --name rag-jina python=3.10

conda activate rag-jina

pip install -r requirements.txt

```

### 2. A simple example(optional)

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# 定义一个函数来对模型的输出进行均值池化:
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# 加载 jina-embeddings-v2-base-zh 的Tokenizer和模型：
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-zh')
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True, torch_dtype=torch.bfloat16)

# 定义一些问题和答案对
knowledge_base = {
    'How is the weather today?': 'The weather is sunny today.',
    '今天天气怎么样?': '今天天气晴朗。',
    'What is the capital of France?': 'The capital of France is Paris.',
    '法国的首都是哪里?': '法国的首都是巴黎。'
}

# 生成知识库的嵌入向量
knowledge_questions = list(knowledge_base.keys())
encoded_knowledge = tokenizer(knowledge_questions, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    knowledge_output = model(**encoded_knowledge)

knowledge_embeddings = mean_pooling(knowledge_output, encoded_knowledge['attention_mask'])
knowledge_embeddings = F.normalize(knowledge_embeddings, p=2, dim=1)

# 定义一个函数来回答问题
def answer_question(question):
    encoded_question = tokenizer(question, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        question_output = model(**encoded_question)

    question_embedding = mean_pooling(question_output, encoded_question['attention_mask'])
    question_embedding = F.normalize(question_embedding, p=2, dim=1)

    similarities = cosine_similarity(question_embedding.numpy(), knowledge_embeddings.numpy())
    most_similar_index = similarities.argmax()

    return knowledge_base[knowledge_questions[most_similar_index]]

# 回答一个问题
question = '今天天气怎么样?'
answer = answer_question(question)
print(f'Question: {question}\nAnswer: {answer}')

```