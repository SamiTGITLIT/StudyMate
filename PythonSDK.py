from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
import lmstudio as lms


embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_function
)


model = lms.llm("mistral-7b-instruct-v0.3")


def answer_query(query):
   
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    
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

    
    response = model.respond(prompt)
    return response


if __name__ == "__main__":
    user_query = "Explain the process of photosynthesis in plants."
    response = answer_query(user_query)
    print(response)
