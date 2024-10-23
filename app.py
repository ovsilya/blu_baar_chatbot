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
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools.retriever import create_retriever_tool


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

store = {}
user_form_trigger_status = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

############################################################
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
folder = "knowledge"
default_replies_eng = load_default_replies(f'{folder}/default_replies_eng.txt')
default_replies_deu = load_default_replies(f'{folder}/default_replies_deu.txt')

############################################################
knowledge_base_eng = read_file(f'{folder}/knowledge_base_eng.txt')
knowledge_base_deu = read_file(f'{folder}/knowledge_base_deu.txt')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits_eng = text_splitter.create_documents([knowledge_base_eng])
splits_deu = text_splitter.create_documents([knowledge_base_deu])
vector_store_eng = Chroma.from_documents(documents=splits_eng, embedding=embeddings)
vector_store_deu = Chroma.from_documents(documents=splits_deu, embedding=embeddings)

retriever_eng = vector_store_eng.as_retriever()
retriever_deu = vector_store_deu.as_retriever()

initial_prompts = {
    "1": "Learn more about services offered by Blu-Baar.",
    "2": "Is parking availbale near the building?",
    "3": "I want to leave my contact information."
}
############################################################

def default_response_eng_tool_func(_):
    return random.choice(default_replies_eng)

def default_response_deu_tool_func(_):
    return random.choice(default_replies_deu)

def lead_form_tool_func(_):
    user_id = getattr(g, 'current_user_id', None)
    if user_id and not user_form_trigger_status.get(user_id):
        cloud_run_url = "https://chatbot-app-94777518696.us-central1.run.app/trigger-lead-form"
        requests.post(cloud_run_url, json={"user_id": user_id})
        user_form_trigger_status[user_id] = True
    return ""

default_response_eng_tool = Tool(
    name="DefaultResponderENG",
    func=default_response_eng_tool_func,
    description="Fallback when no relevant information is found. Use only when user sends message in English.",
    return_direct=True
)

default_response_deu_tool = Tool(
    name="DefaultResponderDEU",
    func=default_response_deu_tool_func,
    description="Fallback when no relevant information is found. Use only when user sends message in German.",
    return_direct=True
)

lead_form_tool = Tool(
    name="LeadForm",
    func=lead_form_tool_func,
    description="Triggers lead form silently.",
    return_direct=False
)

retriever_tool_eng = create_retriever_tool(
    retriever_eng,
    "RetrieverENG",
    description="Retrieves information from the knowledge base. Use only if the user asks questions in English."
)

retriever_tool_deu = create_retriever_tool(
    retriever_deu,
    "RetrieverDEU",
    description="Retrieves information from the knowledge base. Use only if the user asks questions in German."
)

tools = [retriever_tool_eng, 
        retriever_tool_deu, 
        default_response_eng_tool, 
        default_response_deu_tool, 
        lead_form_tool]

llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)
system_prompt = read_file('knowledge/system_prompt.txt')

prompt = PromptTemplate(
        template=system_prompt,
        input_variables=[   "input", 
                            "context",
                            "chat_history"]
    )
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

############################################################
############################################################

# Flask routes
@app.route('/chat', methods=['POST'])
def chat():

    # RECEIVE data from the route: user_message & user_id & InitialPrompt
    data = request.get_json()
    user_message = data.get("message")                  ### 1
    user_id = data.get("user_id")                       ### 2
    initial_prompt = data.get("InitialPrompt", "0")     ### 3

    ################# INITIAL PROMPT ################
    # If user_id is not provided, generate a new one
    if not user_id:
        user_id = str(uuid.uuid4())
    
    if initial_prompt != "0":
        # Use the corresponding initial prompt message
        user_message = initial_prompts.get(initial_prompt, "")
        if not user_message:
            return jsonify({"error": "Invalid InitialPrompt value provided."}), 400
    else:
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

    ################################################
    result = agent_with_chat_history.invoke({"input": user_message}, config={"configurable": {"session_id": "<foo>"}})
    response = result["output"]
    # print("user_id: ",user_id)
    # print("store:",store) 
    return jsonify({"response": response, "user_id": user_id})

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


@app.route('/form-trigger-status', methods=['GET'])
def form_trigger_status():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"message": "No user_id provided."}), 400
    status = user_form_trigger_status.get(user_id, False)
    return jsonify({"user_id": user_id, "form_triggered": status})
############################################################
############################################################
if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=8080, debug=True)
    app.run()
