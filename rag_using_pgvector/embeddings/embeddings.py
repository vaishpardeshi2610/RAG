from sentence_transformers import SentenceTransformer

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Knowledge base (you can replace this with your actual data)
knowledge_base = [
    "An array is a data structure consisting of a collection of elements, each identified by an array index or key.",
    "A function is a block of organized, reusable code that performs a single, related action.",
    "Recursion is a method of solving a problem where the solution depends on solutions to smaller instances of the same problem.",
    "In object-oriented programming, a class is a blueprint for creating objects, providing initial values for state and implementations of behavior.",
    "A loop is used to repeat a block of code as long as a certain condition is met.",
    "In programming, a variable is a storage location paired with an associated symbolic name, which contains some known or unknown quantity."
]

def store_embeddings(conn):
    cur = conn.cursor()
    for doc in knowledge_base:
        embedding = model.encode(doc).tolist()
        cur.execute("INSERT INTO knowledge_base_embeddings (document, embedding) VALUES (%s, %s)", (doc, embedding))

        print("-"*50)
        print(f"Stored embedding for document: {doc}\nEmbedding: {embedding}")
        print("-"*50)
    conn.commit()
    print("-"*50)
    print("Embeddings stored successfully.")
    cur.close()
