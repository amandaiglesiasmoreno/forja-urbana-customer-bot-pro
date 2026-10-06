import os
import logging
import asyncio
from contextlib import asynccontextmanager
from http import HTTPStatus

import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI

from fastapi import FastAPI, Request, Response
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"].strip()
WEBHOOK_URL = os.environ["WEBHOOK_URL"].strip().rstrip("/")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"].strip()
DATABASE_URL = os.environ["DATABASE_URL"].strip()

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
SIMILARITY_THRESHOLD = 0.9

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================================
# 1. Database & History Helpers
# ==========================================

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn

def get_recent_history(session_id: str, limit: int = 4) -> list[dict]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT role, content 
        FROM chat_history 
        WHERE session_id = %s 
        ORDER BY id DESC 
        LIMIT %s
    """, (session_id, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    # Reverse to return older messages first (chronological order)
    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]

def save_message(session_id: str, role: str, content: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO chat_history (session_id, role, content)
        VALUES (%s, %s, %s)
    """, (session_id, role, content))
    conn.commit()
    cur.close()
    conn.close()

# ==========================================
# 2. RAG & LLM Logic
# ==========================================

def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding

def retrieve_context(query: str, k: int = 5) -> list[str]:
    embedding = get_embedding(query)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT content, embedding <=> %s::vector AS distance
        FROM documents
        ORDER BY distance ASC
        LIMIT %s;
        """,
        (embedding, k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    logger.info(f"=== RAG DEBUG: Query -> '{query}' ===")
    for idx, (content, distance) in enumerate(rows, 1):
        logger.info(f"Chunk #{idx} | Distance: {distance:.4f} | Content preview: {content[:100]}...")

    return [content for content, distance in rows]

def reformulate_query(history: list[dict], current_query: str) -> str:
    if not history:
        return current_query
        
    system_prompt = (
        "You are an expert query reformulation assistant for a RAG system. "
        "Your task is to analyze the chat history and rewrite the user's latest message "
        "into a complete, standalone query that requires no prior context to be understood.\n\n"
        "Apply these core strategies where necessary:\n"
        "1. Resolve Pronouns: Replace ambiguous pronouns (e.g., it, he, them, that) with the specific entities mentioned previously.\n"
        "2. Clarify Implicit Comparisons: If the user asks about a new element in relation to a previously discussed one, explicitly name both elements to form a comparative query.\n"
        "3. Complete Elliptical Queries: Fill in missing subjects, verbs, or context that the user omitted assuming you would remember.\n\n"
        "STRICT CONSTRAINTS:\n"
        "- If the latest message is a greeting, conversational filler, or already fully self-contained, return it EXACTLY as is.\n"
        "- DO NOT attempt to answer the question.\n"
        "- OUTPUT ONLY the reformulated query text. Do not include quotes, prefixes like 'Rewritten query:', or explanations."
    )

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": f"Latest message: {current_query}"})
    
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

async def rag_answer(session_id: str, original_query: str) -> str:
    # 1. Fetch history & Reformulate
    history = get_recent_history(session_id, limit=4)
    standalone_query = reformulate_query(history, original_query)
    logger.info(f"Reformulated Query: {standalone_query}")

    # 2. Retrieve Context based on standalone query
    context_chunks = retrieve_context(standalone_query)
    if not context_chunks:
        logger.warning("RAG WARNING: context_chunks is empty.")
        return "I don't have enough information in my knowledge base to answer that."

    context = "\n\n---\n\n".join(context_chunks)
    
    # 3. Build final conversational prompt
    messages = [
        {"role": "system", "content": f"Answer the user's question using ONLY the provided context.\n\nContext:\n{context}"}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": original_query})

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0,
    )
    
    answer = response.choices[0].message.content
    return answer

# ==========================================
# 3. Telegram Bot Handlers
# ==========================================

ptb_app = (
    Application.builder()
    .updater(None)
    .token(BOT_TOKEN)
    .read_timeout(7)
    .get_updates_read_timeout(42)
    .build()
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_text(
        f"Hello {user.first_name}! 👋\n"
        "Ask me anything about my knowledge base."
    )

async def rag_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.message.text
    session_id = str(update.effective_user.id)
    
    logger.info(f"Received message from user {session_id}: {query}")
    await update.message.chat.send_action(action="typing")
    
    # Save user message
    save_message(session_id, "user", query)
    
    # Generate Answer
    answer = await rag_answer(session_id, query)
    
    # Save bot answer
    save_message(session_id, "assistant", answer)
    
    await update.message.reply_text(answer)

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Unknown command. Try /start")

ptb_app.add_handler(CommandHandler("start", start))
ptb_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, rag_handler))
ptb_app.add_handler(MessageHandler(filters.COMMAND, unknown))

# ==========================================
# 4. Background TTL Task & FastAPI Setup
# ==========================================

async def run_daily_cleanup():
    """Background asyncio task to clean up DB history older than 7 days."""
    while True:
        try:
            # Run sync DB call in a thread to avoid blocking the async event loop
            def _clean():
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("DELETE FROM chat_history WHERE created_at < NOW() - INTERVAL '7 days';")
                deleted = cur.rowcount
                conn.commit()
                cur.close()
                conn.close()
                return deleted
            
            deleted_rows = await asyncio.to_thread(_clean)
            if deleted_rows > 0:
                logger.info(f"TTL Cleanup: Purged {deleted_rows} old chat messages.")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            
        await asyncio.sleep(86400) # Sleep for 24 hours

@asynccontextmanager
async def lifespan(_: FastAPI):
    # 1. Start background cleanup task
    cleanup_task = asyncio.create_task(run_daily_cleanup())
    
    # 2. Setup Telegram Webhook
    webhook_endpoint = f"{WEBHOOK_URL}/telegram"
    await ptb_app.bot.setWebhook(webhook_endpoint)
    logger.info(f"Webhook set: {webhook_endpoint}")
    
    # 3. Start bot
    async with ptb_app:
        await ptb_app.start()
        yield
        await ptb_app.stop()
        
    # 4. Teardown
    cleanup_task.cancel()
    await ptb_app.bot.deleteWebhook()
    logger.info("Webhook removed.")

app = FastAPI(lifespan=lifespan)

@app.post("/telegram")
async def telegram_webhook(request: Request) -> Response:
    data = await request.json()
    update = Update.de_json(data, ptb_app.bot)
    await ptb_app.process_update(update)
    return Response(status_code=HTTPStatus.OK)

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
