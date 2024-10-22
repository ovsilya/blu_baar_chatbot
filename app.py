import os
import random
import uuid
import logging
import requests
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from nltk.tokenize import sent_tokenize
from logging.handlers import RotatingFileHandler
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import NLTKTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent

app = Flask(__name__)
CORS(app)

############################################################
# Logger setup
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chatbot_interactions.log')
handler = RotatingFileHandler(log_file_path, maxBytes=5 * 1024 * 1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger('chatbot')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

############################################################
# Load environment variables and data files
OPENAI_API_KEY = "sk-proj-KGUXTSrwq2m2wueI4JwKT3BlbkFJHcNZecVcvLRSNxQgMjkM"

def read_file(file_name):
    """Helper function to read a file's content."""
    with open(file_name, 'r') as file:
        return file.read()

def load_default_replies(file_path):
    """Load default responses for fallback when no relevant documents are found."""
    replies = [line.strip() for line in read_file(file_path).splitlines() if line.strip()]
    return replies

def split_into_chunks(text, language='english', max_chunk_size=500):

    sentences = sent_tokenize(text, language=language)  # Use the language for tokenization
    chunks = []
    current_chunk = ''
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += ' ' + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def initialize_vector_store(file_path, embeddings, language='english'):

    if os.path.exists(f"faiss_index_{os.path.basename(file_path).split('.')[0]}"):
        # Load from the saved FAISS index if it exists
        vector_store = FAISS.load_local(f"faiss_index_{os.path.basename(file_path).split('.')[0]}", embeddings, allow_dangerous_deserialization=True)
        print(f"Loaded FAISS index for {file_path} from disk.")
    else:
        # If no saved FAISS index, process the file and create a new index
        text_data = load_chunks(file_path)
        chunks = split_into_chunks(text_data, language=language, max_chunk_size=500)
        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local(f"faiss_index_{os.path.basename(file_path).split('.')[0]}")
        print(f"Created and saved FAISS index for {file_path}.")
    
    return vector_store

############################################################
# Initialize embeddings and FAISS vector store
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# Initialize vector stores for English and German knowledge bases
vector_store_eng = initialize_vector_store('knowledge_base_eng.txt', embeddings, language='english')
vector_store_deu = initialize_vector_store('knowledge_base_deu.txt', embeddings, language='german')

default_replies_eng = load_default_replies('default_replies_eng.txt')
default_replies_deu = load_default_replies('default_replies_deu.txt')

############################################################
# Define tools for agent
def retriever_tool_func(query):
    docs = vector_store.similarity_search(query)
    if not docs:
        return default_response_tool_func(None)
    return "\n".join([doc.page_content for doc in docs])

def default_response_eng_tool_func(_):
    return random.choice(default_replies_eng)

def lead_form_tool_func(_):
    user_id = getattr(g, 'current_user_id', None)
    if user_id and not user_form_trigger_status.get(user_id):
        cloud_run_url = "https://chatbot-app-94777518696.us-central1.run.app/trigger-lead-form"
        requests.post(cloud_run_url, json={"user_id": user_id})
        user_form_trigger_status[user_id] = True
    return ""

lead_form_tool = Tool(
    name="LeadForm",
    func=lead_form_tool_func,
    description="Triggers lead form silently.",
    return_direct=False
)

retriever_tool = Tool(
    name="Retriever",
    func=retriever_tool_func,
    description="Retrieves information from the knowledge base."
)

default_response_tool = Tool(
    name="DefaultResponder",
    func=default_response_eng_tool_func,
    description="Fallback when no relevant information is found.",
    return_direct=True
)

tools = [retriever_tool, default_response_tool, lead_form_tool]
############################################################

# Chat model and agent setup
llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)

def create_langchain_agent(memory):
    """Create the agent using LangChain's tool calling method."""
    system_prompt_text = read_file('system_prompt.txt')
    prompt = PromptTemplate(
        template=system_prompt_text,
        input_variables=["input", "context", "chat_history", "tools", "tool_names", "agent_scratchpad"]
    )
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )
    return agent_executor

# Conversation and interaction management
conversation_history = {}
user_form_trigger_status = {}

############################################################
############################################################
def interact_with_user(user_message, user_id):
    """Handle user interaction by retrieving memory, running the agent, and managing lead form triggers."""
    if user_id not in conversation_history:
        conversation_history[user_id] = {
            "memory": ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                input_key="input",
                output_key="output"
            ),
            "interaction_count": 0
        }

    memory = conversation_history[user_id]["memory"]
    interaction_count = conversation_history[user_id]["interaction_count"] + 1
    conversation_history[user_id]["interaction_count"] = interaction_count

    lead_form_triggered = user_form_trigger_status.get(user_id, False)

    if interaction_count >= 7 and not lead_form_triggered:
        requests.post("https://chatbot-app-94777518696.us-central1.run.app/trigger-lead-form", json={"user_id": user_id})
        user_form_trigger_status[user_id] = True
        logger.info(f"User ID: {user_id} - Lead form triggered after 7 interactions.")

    g.current_user_id = user_id
    agent_executor = create_langchain_agent(memory)
    docs = vector_store.similarity_search(user_message)
    context = "\n".join([doc.page_content for doc in docs])

    if not docs:
        logger.info(f"User ID: {user_id} - Default response used due to lack of relevant docs.")
        return default_response_tool_func(None), lead_form_triggered

    try:
        result = agent_executor.invoke({"input": user_message, "context": context})
        response = result["output"]
        intermediate_steps = result["intermediate_steps"]

        tools_used = [step[0].tool for step in intermediate_steps if step[0].tool]
        logger.info(f"User ID: {user_id} - Tools used: {tools_used}")

        if 'LeadForm' in tools_used and not lead_form_triggered:
            requests.post("https://chatbot-app-94777518696.us-central1.run.app/trigger-lead-form", json={"user_id": user_id})
            user_form_trigger_status[user_id] = True
            lead_form_triggered = True
            logger.info(f"User ID: {user_id} - Lead form triggered by LeadForm tool.")

        return response, lead_form_triggered

    except Exception as e:
        logger.error(f"User ID: {user_id} - Error: {str(e)}")
        return "An error occurred.", lead_form_triggered

# Flask routes
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message")
    user_id = data.get("user_id", str(uuid.uuid4()))
    initial_prompt = data.get("InitialPrompt", "0")

    if initial_prompt != "0":
        user_message = initial_prompts.get(initial_prompt, "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    logger.info(f"User ID: {user_id} - Received message: '{user_message}'")
    response, lead_form_triggered = interact_with_user(user_message, user_id)
    return jsonify({"response": response, "user_id": user_id, "lead_form_triggered": lead_form_triggered})

@app.route('/trigger-lead-form', methods=['POST'])
def trigger_lead_form():
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"message": "No user_id provided."}), 400

    if user_form_trigger_status.get(user_id):
        return jsonify({"message": "Lead form already triggered for this user."}), 200

    user_form_trigger_status[user_id] = True
    logger.info(f"User ID: {user_id} - Lead form sent to user.")
    return jsonify({"action": "show_lead_form", "user_id": user_id})

############################################################
############################################################
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
    # app.run(debug=True)
