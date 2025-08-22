from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
import os

from dotenv import load_dotenv
load_dotenv()

raw_documents = TextLoader("tagged_descriptions.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("QWEN_API_KEY"),
)

# 设置持久化目录
persist_directory = "./db_books"  # 本地存储路径

# 初始化 Chroma 数据库，启用持久化
db_books = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_directory
)

batch_size = 50

for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    try:
        db_books.add_documents(batch)
        print(f"处理第 {i // batch_size + 1} 批成功")
    except Exception as e:
        print(f"处理第 {i // batch_size + 1} 批失败: {e}")
        break

if db_books:
    print(f"成功存储 {db_books._collection.count()} 条记录")