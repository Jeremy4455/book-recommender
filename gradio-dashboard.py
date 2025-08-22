import os
import textwrap

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma

import gradio as gr
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "CoverNotAvailable.jpg",
    books["large_thumbnail"],
)

embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("QWEN_API_KEY"),
)

# 加载本地的Chroma数据库
persist_directory = "./db_books"
db_books = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_directory
)

print(f"加载了 {db_books._collection.count()} 条记录")


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    books_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        books_recs = books_recs[books_recs["simple_categories"] == category].head(final_top_k)
    else:
        books_recs = books_recs.head(final_top_k)

    if tone == "Happy" or tone == "快乐":
        books_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising" or tone == "惊喜":
        books_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry" or tone == "愤怒":
        books_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful" or tone == "悬疑":
        books_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad" or tone == "难过":
        books_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return books_recs


def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        # Wrap description at approximately 100 characters
        wrapped_description = textwrap.fill(description, width=100, max_lines=2, placeholder="...")

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {wrapped_description}"
        results.append((row["large_thumbnail"], caption))
    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad", "快乐", "惊喜", "愤怒", "悬疑", "难过"]

custom_css = """
    .gr-button {
        background-color: #1e40af !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        transition: background-color 0.3s ease !important;
    }
    .gr-button:hover {
        background-color: #1e3a8a !important;
    }
    .gr-textbox, .gr-dropdown {
        border-radius: 8px !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    .gr-gallery {
        background-color: #f9fafb !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    .gr-markdown h2 {
        color: #1e40af !important;
        font-weight: 600 !important;
        margin-bottom: 1.5rem !important;
    }
    .container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding: 20px !important;
    }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as dashboard:
    gr.Markdown(
        """
        ## 📚 书籍推荐系统
        欢迎体验智能书籍推荐！输入您的阅读偏好，选择类别和情感基调，我们将为您推荐最合适的书籍。
        """,
        elem_classes=["container"]
    )

    with gr.Row(variant="panel", equal_height=True):
        user_query = gr.Textbox(
            label="📝 您想读什么样的书？",
            placeholder="例如：一个跌宕起伏的冒险故事",
            lines=2,
            elem_classes=["gr-textbox"]
        )
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="📚 书籍类别",
            value="All",
            elem_classes=["gr-dropdown"]
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="🎭 情感基调",
            value="All",
            elem_classes=["gr-dropdown"]
        )
        submit_button = gr.Button("🔍 搜索推荐", variant="primary")

    gr.Markdown("## 📖 推荐书单", elem_classes=["container"])
    output = gr.Gallery(
        label="书籍推荐",
        columns=4,
        rows=2,
        object_fit="cover",
        height="auto",
        elem_classes=["gr-gallery"],
        show_download_button=False,
        show_share_button=False
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output  # Show loading animation
    )

if __name__ == "__main__":
    dashboard.launch()
