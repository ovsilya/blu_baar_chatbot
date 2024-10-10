from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import NLTKTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from nltk.tokenize import sent_tokenize
import os
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid  # For generating unique user IDs

app = Flask(__name__)
CORS(app)


def load_chunks():
    with open('chunks_for_embeddings.txt', 'r') as file:
        return file.read()
    

def split_into_chunks(text, max_chunk_size=500):
    sentences = sent_tokenize(text, language='english')
    chunks = []
    current_chunk = ''
    
    for sentence in sentences:
        # Check if adding the sentence exceeds the max chunk size
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += ' ' + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks





text_data = load_chunks()
chunks = split_into_chunks(text_data, max_chunk_size=500)

print('Chunks: ',chunks[:2])


# Initialize OpenAI embeddings
OPENAI_API_KEY = os.environ.get('NAVAICGKEY')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load or create FAISS index
if os.path.exists("faiss_index"):
    # vector_store = FAISS.load_local("faiss_index", embeddings)
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    print("Loaded FAISS index from disk.")
else:
    # Assuming 'chunks' is a list of your text data split into chunks
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    vector_store.save_local("faiss_index")
    print("Created and saved FAISS index.")



llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory,
    verbose=True
)


conversation_history = {}

def interact_with_user(user_message, user_id):
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    
    chat_history = conversation_history[user_id]
    
    result = qa({"question": user_message, "chat_history": chat_history})
    response = result["answer"]
    
    # Update conversation history
    chat_history.append({"user": user_message, "assistant": response})
    
    # Optionally, check for lead generation after a certain number of interactions
    if len(chat_history) >= 3:
        response += "\n\nIt seems you're interested! Would you like to leave your contact details?"
        response += " Please provide your name, email, and phone number."

    conversation_history[user_id] = chat_history

    return response


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message")
    user_id = data.get("user_id")
    
    if not user_id:
        # Generate a new user_id if not provided
        user_id = str(uuid.uuid4())
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    response = interact_with_user(user_message, user_id)
    
    return jsonify({"response": response, "user_id": user_id})

if __name__ == '__main__':
    app.run(debug=True)
