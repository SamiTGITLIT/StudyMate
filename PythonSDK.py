# Import necessary libraries
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
import lmstudio as lms

# Initialize the embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the existing Chroma vector store from disk
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_function
)

# Initialize the LM Studio model
model = lms.llm("mistral-7b-instruct-v0.3")

# Define a function to answer queries using RAG
def answer_query(query):
    # Retrieve relevant documents from the vector store
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Construct the prompt with a system message
    prompt = f"""
<s>[INST]
You are a knowledgeable and engaging biology instructor, adept at explaining complex biological concepts in a clear and accessible manner. Your explanations are structured, incorporating real-world examples and analogies to facilitate understanding.

Using the following context, answer the question below:

Context:
{context}

Question:
{query}
[/INST]
"""

    # Generate the response using the LM Studio model
    response = model.respond(prompt)
    return response

# Example usage
if __name__ == "__main__":
    user_query = "Explain the process of photosynthesis in plants."
    response = answer_query(user_query)
    print(response)
