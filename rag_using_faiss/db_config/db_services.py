import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DatabaseConfig:
    def __init__(self):
        self.dbname = os.getenv("DB_NAME")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.host = os.getenv("DB_HOST", 'localhost')
        self.port = os.getenv("DB_PORT", '5432')
        self.connection = None
        self.cursor = None

    def connect(self):
        """Establish a connection to the PostgreSQL database."""
        self.connection = psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )
        self.cursor = self.connection.cursor()

    def create_table(self):
        """Create a table for storing document embeddings if it doesn't exist."""
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_embeddings (
            id SERIAL PRIMARY KEY,
            document TEXT,
            embedding FLOAT8[]
        );
        """)
        self.connection.commit()

    def insert_embeddings(self, documents, embeddings):
        """Insert multiple documents and their embeddings into the database."""
        for document, embedding in zip(documents, embeddings):
            self.cursor.execute(""" 
            INSERT INTO document_embeddings (document, embedding) VALUES (%s, %s);
            """, (document, embedding))
        self.connection.commit()

    def close(self):
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
