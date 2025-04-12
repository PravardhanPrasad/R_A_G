from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function  # Ensure this exists

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app, resources={r"/query": {"origins": "*"}})  # Enable CORS for API routes

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Initialize the embedding function and Chroma database
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

@app.route("/")
def home():
    """Serve the frontend HTML."""
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.json
        query_text = data.get("query_text")

        if not query_text:
            return jsonify({"error": "No query text provided"}), 400

        results = db.similarity_search_with_score(query_text, k=1)

        if results:
            best_doc, score = results[0]
            context_text = getattr(best_doc, "page_content", "No content found.")  # Safe access
            accuracy_score = max(0.0, 1 - score)  # Ensure score stays within range
        else:
            context_text = "No relevant data found."
            accuracy_score = 0.0

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = OllamaLLM(model="llama3.2")

        general_response = model.invoke(query_text).strip()
        rag_response = model.invoke(prompt).strip()

        return jsonify({
            "general_response": general_response,
            "rag_response": rag_response,
            "exact_chunk": context_text.strip(),
            "accuracy_score": accuracy_score
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
