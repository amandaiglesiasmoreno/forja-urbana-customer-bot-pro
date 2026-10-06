import os
import psycopg2

DATABASE_URL = os.environ["DATABASE_URL"]

def main():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Enable pgvector
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 1. Documents table for embeddings
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR(1536)
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS documents_embedding_idx
        ON documents USING hnsw (embedding vector_cosine_ops);
    """)

    # 2. Chat history table for conversational RAG
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Indexes to speed up history retrieval and TTL cleanup
    cur.execute("""
        CREATE INDEX IF NOT EXISTS chat_history_session_idx
        ON chat_history (session_id);
    """)
    
    cur.execute("""
        CREATE INDEX IF NOT EXISTS chat_history_created_at_idx
        ON chat_history (created_at);
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Database configured successfully with documents and chat_history tables.")

if __name__ == "__main__":
    main()