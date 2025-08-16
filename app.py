import numpy as np
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from transformers import pipeline

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings



books= pd.read_csv("books_with_emotions.csv")
books['large_thumbnail']= books['thumbnail']+"&fife=w800"
books['large_thumbnail']= np.where(
    books['large_thumbnail'].isna(), 'cover not available.jpg', books['large_thumbnail']
)

# Load BAAI embedding model with LangChain wrapper
embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

db_books = Chroma(
    persist_directory="./db_books" ,
    embedding_function=embedding_function
)

def retrieve_recommendations(
        query: str, 
        category: str= None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
        ):
    """Retrieve book recommendations based on a query."""
    recs = db_books.similarity_search(query, k=initial_top_k)

    books_list= [int(rec.page_content.strip('"').split()[0]) for rec in recs]

    books_recs = books[books['isbn13'].isin(books_list)].head(final_top_k)

    if category != 'All':
        books_recs = books_recs[books_recs['simple_categories'] == category][:final_top_k]
    else:
        books_recs = books_recs.head(final_top_k)

    if tone == 'Happy':
        books_recs.sort_values(by='joy', ascending=False, inplace=True)

    elif tone == 'Surprising':
        books_recs.sort_values(by='surprise', ascending=False, inplace=True)

    elif tone == 'Sad':
        books_recs.sort_values(by='sadness', ascending=False, inplace=True)

    elif tone == 'Angry':
        books_recs.sort_values(by='anger', ascending=False, inplace=True)   
    
    elif tone == 'Suspenseful':
        books_recs.sort_values(by='fear', ascending=False, inplace=True)

    return books_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    """Function to recommend books based on user input."""
    recommendations = retrieve_recommendations(
        query=query,
        category=category,
        tone=tone
    )
    
    results = []
    for _, row in recommendations.iterrows():

        description= row['description']
        truncated_desc_split = description.split()
        truncated_desc = ' '.join(truncated_desc_split[:30]) + '...'

        authors_split = row['authors'].split(';')
        if len(authors_split) == 2:
            authors = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors = f"{'; '.join(authors_split[:-1])}, and {authors_split[-1]}"   

        else:
            authors = row['authors']
        
        caption= f"{row['title']} by {authors}: {truncated_desc}"

        results.append((row['large_thumbnail'], caption))

    return results
    

categories = ['All'] + sorted(books['simple categories'].unique())
tones= ['All', 'Happy', 'Surprising', 'Sad', 'Angry', 'Suspenseful']

with gr.Blocks(theme= gr.themes.Glass()) as dashboard:
    gr.Markdown(
        """
        # Book Recommender
        Find your next favorite book based on your mood and interests!
        """
    )

    with gr.Row():
        user_query = gr.Textbox(
            label="What kind of book are you looking for?",
            placeholder="e.g. 'I want a happy book about friendship'"
        )

        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select a Category",
            value='All'
        )

        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select a Tone",
            value='All'
        )

        submit_button = gr.Button("Get Recommendations")

    gr.Markdown("## Recommendations")
    output= gr.Gallery(label="Recommended Books", columns=8, rows= 2)

    submit_button.click(fn= recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)
    

if __name__ == "__main__":
    dashboard.launch()



