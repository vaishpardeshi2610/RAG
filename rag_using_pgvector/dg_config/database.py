import psycopg2

def ensure_pgvector_extension(conn):
    cur = conn.cursor()
    # Check if the pgvector extension exists
    cur.execute("""
    SELECT * FROM pg_extension WHERE extname = 'vector';
    """)
    if not cur.fetchone():
        print("pgvector extension not found. Installing it now...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        print("pgvector extension installed successfully.")
    else:
        print("pgvector extension is already installed.")
    cur.close()

def create_embeddings_table(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS knowledge_base_embeddings (
        id SERIAL PRIMARY KEY,
        document TEXT,
        embedding VECTOR(384)  -- Dimension of 'all-MiniLM-v2'
    );
    """)
    conn.commit()
    cur.close()

def close_connection(conn):
    conn.close()
