import os
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from pypdf import PdfReader

DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DOCS_PATH = "./docs"
EMBEDDING_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=OPENAI_API_KEY)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def load_documents() -> list[tuple[str, str]]:
    docs = []
    if not os.path.exists(DOCS_PATH):
        print(f"Creating directory {DOCS_PATH}")
        os.makedirs(DOCS_PATH)
        return docs

    for filename in os.listdir(DOCS_PATH):
        path = os.path.join(DOCS_PATH, filename)
        if filename.endswith(".pdf"):
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            continue
        docs.append((filename, text))
    return docs

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding

def main():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    cur = conn.cursor()

    # Clear old documents to avoid duplicates on re-runs
    cur.execute("TRUNCATE TABLE documents RESTART IDENTITY;")

    documents = load_documents()
    total_chunks = 0

    for filename, text in documents:
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            cur.execute(
                """
                INSERT INTO documents (source, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (filename, i, chunk, embedding),
            )
            total_chunks += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"Indexed {total_chunks} chunks from {len(documents)} documents.")

if __name__ == "__main__":
    main()