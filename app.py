from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompts import (
    MEDICAL_CHATBOT_PROMPT,
    DIAGNOSTIC_PROMPT,
    MEDICAL_DISCLAIMER
)


app = Flask(__name__)

load_dotenv()

chatbot_chain = None
diagnostic_chain = None

def initialize_chatbot():
    global chatbot_chain, diagnostic_chain

    embeddings = download_hugging_face_embeddings()

    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index_name = "medical-chatbot"
    index = pc.Index(index_name)

    vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
    
    llm = ChatNVIDIA(
            model="meta/llama-3.1-70b-instruct",
            api_key="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",
            temperature=0.7,
            max_tokens=1024,
        )
    
    question_answer_chain = create_stuff_documents_chain(llm, MEDICAL_CHATBOT_PROMPT)
    chatbot_chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 3}),  # 3 documents
        question_answer_chain
    )

    diagnostic_answer_chain = create_stuff_documents_chain(llm, DIAGNOSTIC_PROMPT)
    diagnostic_chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 5}),  # 5 documents pour plus de contexte
        diagnostic_answer_chain
    )

@app.route('/')
def home():
    """Page d'accueil du chatbot."""
    return render_template('chat.html', disclaimer=MEDICAL_DISCLAIMER)

@app.route('/ask', methods=['POST'])
def ask():

    """Endpoint principal pour questions et diagnostic."""

    data = request.get_json()
    question = data.get('question', '').strip()
    mode = data.get('mode', 'chat')  # 'chat' ou 'diagnose'
    
    if not question:
        return jsonify({'error': 'Question vide', 'success': False}), 400
    
    # Choisir la chaîne selon le mode
    chain = diagnostic_chain if mode == 'diagnose' else chatbot_chain
    response = chain.invoke({"input": question})
    
    # Extraire les sources
    sources = list(set([
        doc.metadata.get('source', 'Unknown')
        for doc in response.get('context', [])
    ]))

    return jsonify({
        'success': True,
        'answer': response['answer'],
        'sources': sources,
        'mode': mode
    })

@app.route('/health')
def health():
    """Vérifie l'état de l'application."""
    return jsonify({
        'status': 'healthy' if chatbot_chain else 'not_ready'
    })

if __name__ == '__main__':
    initialize_chatbot()
    print("🚀 Serveur Flask démarré sur http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=True)