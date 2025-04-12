import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)


    results = db.similarity_search_with_score(query_text, k=5)
    if not results:
        print("No relevant chunks found in the database.")
        return


    exact_chunk, exact_chunk_score = results[0]


    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)


    model = Ollama(model="llama3.2")
    general_response = model.invoke(query_text)


    rag_response = model.invoke(prompt)


    print("\n===== General Llama2 Response =====")
    print(general_response)

    print("\n===== RAG Response (Based on Retrieved Context) =====")
    print(rag_response)

    print("\n===== Exact Chunk from Database =====")
    print(f"Content: {exact_chunk.page_content}")
    print(f"Metadata: {exact_chunk.metadata}")
    print(f"Score: {exact_chunk_score}")

    return {
        "general_response": general_response,
        "rag_response": rag_response,
        "exact_chunk": exact_chunk.page_content,
    }


if __name__ == "__main__":
    main()
