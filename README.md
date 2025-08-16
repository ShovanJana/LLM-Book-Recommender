# Book Recommender

Find your next favorite book based on your mood and interests!  
This project uses Gradio for the user interface and LangChain for semantic search over a curated book dataset.

## Features

- Search for books by query, category, and emotional tone
- Recommendations powered by vector search and emotion tagging
- Interactive web dashboard

## How It Works

- Uses LangChain and HuggingFace embeddings for semantic search
- Book data is vectorized and stored in a local Chroma database
- User queries are matched to books by similarity and emotional tone

## Getting Started

### Prerequisites

- Python 3.10 or higher

### Installation

1. **Clone the repository:**
   ```bash
    git clone https://github.com/yourusername/LLM-Book-Recommender.git
    cd LLM-Book-Recommender
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Make sure the following files are present:**
   - `books_with_emotions.csv`
   - `cover not available.jpg`
   - `db_books/` directory (vector database)

### Running the App

```bash
python gradio-dashboard.py
```

The Gradio dashboard will launch in your browser.

## Project Structure

```
LLM Book Recommender/
├── gradio-dashboard.py
├── requirements.txt
├── books_with_emotions.csv
├── db_books/
├── cover not available.jpg
├── *.ipynb (notebooks for data processing)
```

## License

MIT License

## Acknowledgements

- [Gradio](https://gradio.app/)
- [LangChain](https://langchain.dev/)
- [Hugging Face](https://huggingface.co/)
