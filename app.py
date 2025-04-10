import os
import random
import uuid
import logging
import requests
import re
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
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
import csv
from functools import partial
from flask_socketio import SocketIO, emit
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

import eventlet
import eventlet.wsgi

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, Personalization

app = Flask(__name__)
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*")

############################################################
# Logger setup
# log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chatbot_interactions.log')
# handler = RotatingFileHandler(log_file_path, maxBytes=5 * 1024 * 1024, backupCount=5)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)

# logger = logging.getLogger('chatbot')
# logger.setLevel(logging.INFO)
# logger.addHandler(handler)

############################################################

def read_file(file_name):
    """Helper function to read a file's content."""
    with open(file_name, 'r') as file:
        return file.read()

def load_default_replies(file_path):
    """Load default responses for fallback when no relevant documents are found."""
    replies = [line.strip() for line in read_file(file_path).splitlines() if line.strip()]
    return replies

def log_chat_history(session_id: str):
    history = get_session_history(session_id)
    # logger.info(f"Chat history for session {session_id}: {history.messages}")
    # Alternatively, print the history:
    # print(f"Chat history for session {session_id}: {history.messages}")

def remove_double_asterisks(text): 

    pattern = r'\*\*(.*?)\*\*'
    return re.sub(pattern, r'\1', text)

def replace_sharp_s(text):
    return text.replace("ß", "ss")



def detect_language(text):
    try:
        lang = detect(text)
        return 'ENG' if lang == 'en' else 'DEU' if lang == 'de' else None
    except LangDetectException:
        return None

# Fetch data from Google Sheets and create retrievers for PDF descriptions
def fetch_google_sheet_data():
    sheet_url =os.getenv('GOOGLE_PATH')
    response = requests.get(sheet_url)
    response.raise_for_status()
    data = response.content.decode('utf-8')
    reader = csv.reader(data.splitlines())
    records = [row for row in reader]
    return records

def pdf_retriever_tool_func(query, retriever, language):
    docs = retriever.get_relevant_documents(query)
    if docs:
        doc = docs[0] 
        description = doc.page_content
        link = doc.metadata.get('link', 'No link available.' if language == 'ENG' else 'Kein Link verfügbar.')
        doc_name = doc.metadata.get('name', 'Relevant Document' if language == 'ENG' else 'Einschlägiges Dokument')
        # Ensure the clickable format with relevant word in square brackets followed by link in parentheses
        return f"[{doc_name}]({link})\n\n{description}"
    else:
        return "No relevant document found." if language == 'ENG' else "Keine relevanten Dokumente gefunden."


store = {}
user_form_trigger_status = {}  # Tracks if lead form has been shown for each user
user_interaction_count = {}    # Tracks the number of interactions per user
message_language = {}         # Tracks language of the user 

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

############################################################
embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
folder = "knowledge"
default_replies_eng = load_default_replies(f'{folder}/default_replies_eng.txt')
default_replies_deu = load_default_replies(f'{folder}/default_replies_deu.txt')

############################################################
# Read the glossary files
glossary_eng = read_file(f'{folder}/glossary_eng.txt')
glossary_deu = read_file(f'{folder}/glossary_deu.txt')

knowledge_base_eng = read_file(f'{folder}/knowledge_base_eng.txt')
questions_eng = read_file(f'{folder}/questions_answers_eng.txt')
# full_information_eng = knowledge_base_eng + "\n\n" + questions_eng

knowledge_base_deu = read_file(f'{folder}/knowledge_base_deu.txt')
questions_deu = read_file(f'{folder}/questions_answers_deu.txt')
# full_information_deu = knowledge_base_deu + "\n\n" + questions_deu

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# create the glossary text retrievers
splits_glossary_eng = text_splitter.create_documents([glossary_eng])
splits_glossary_deu = text_splitter.create_documents([glossary_deu])
vector_store_glossary_eng = Chroma.from_documents(documents=splits_glossary_eng, embedding=embeddings,collection_name="glossary_eng")
vector_store_glossary_deu = Chroma.from_documents(documents=splits_glossary_deu,embedding=embeddings,collection_name="glossary_deu")
retriever_glossary_eng = vector_store_glossary_eng.as_retriever()
retriever_glossary_deu = vector_store_glossary_deu.as_retriever()

# create Knowledge base retrievers
splits_eng = text_splitter.create_documents([knowledge_base_eng])
splits_deu = text_splitter.create_documents([knowledge_base_deu])
vector_store_eng = Chroma.from_documents(documents=splits_eng, embedding=embeddings)
vector_store_deu = Chroma.from_documents(documents=splits_deu, embedding=embeddings)
retriever_eng = vector_store_eng.as_retriever()
retriever_deu = vector_store_deu.as_retriever()

# create questions-answers retrievers
splits_eng_qa = text_splitter.create_documents([questions_eng])
splits_deu_qa = text_splitter.create_documents([questions_deu])
vector_store_eng_qa = Chroma.from_documents(documents=splits_eng_qa, embedding=embeddings)
vector_store_deu_qa = Chroma.from_documents(documents=splits_deu_qa, embedding=embeddings)
retriever_eng_qa = vector_store_eng_qa.as_retriever()
retriever_deu_qa = vector_store_deu_qa.as_retriever()

# Create BM25 retrievers
bm25_retriever_eng = BM25Retriever.from_documents(splits_eng)
bm25_retriever_deu = BM25Retriever.from_documents(splits_deu)

# Create Ensemble retrievers
ensemble_retriever_eng = EnsembleRetriever(retrievers=[retriever_eng, bm25_retriever_eng, retriever_eng_qa])
ensemble_retriever_deu = EnsembleRetriever(retrievers=[retriever_deu, bm25_retriever_deu, retriever_deu_qa])


# Working with the cvs files with links to the PDF
# take csv file from google drive
pdf_records = fetch_google_sheet_data()

# Assuming the columns are: 0 - name, 1 - language, 2 - description, 3 - link
pdf_records_eng = [rec for rec in pdf_records if rec[1].strip().lower() == 'eng']
pdf_records_deu = [rec for rec in pdf_records if rec[1].strip().lower() == 'deu']


# Create documents from descriptions
documents_eng = [
    Document(
        page_content=rec[2], 
        metadata={'name': rec[0], 'link': rec[3]}
    ) for rec in pdf_records_eng
]

documents_deu = [
    Document(
        page_content=rec[2],
        metadata={'name': rec[0], 'link': rec[3]}
    ) for rec in pdf_records_deu
]

# Create vector stores for PDF descriptions
vector_store_pdfs_eng = Chroma.from_documents(
    documents=documents_eng,
    embedding=embeddings,
    collection_name="pdfs_eng"
)

vector_store_pdfs_deu = Chroma.from_documents(
    documents=documents_deu,
    embedding=embeddings,
    collection_name="pdfs_deu"
)

# Create retrievers for PDFs
retriever_pdfs_eng = vector_store_pdfs_eng.as_retriever()
retriever_pdfs_deu = vector_store_pdfs_deu.as_retriever()

# It is for using function pdf_retriever_tool_func twice with language parameter
pdf_retriever_eng_tool_func = partial(pdf_retriever_tool_func, retriever=retriever_pdfs_eng, language='ENG')
pdf_retriever_deu_tool_func = partial(pdf_retriever_tool_func, retriever=retriever_pdfs_deu, language='DEU')

initial_prompts = {
    # "1": "Erfahren Sie mehr über die von Blu-Baar angebotenen Dienstleistungen.",
    # "2": "Gibt es Parkplätze in der Nähe oder im Gebäude?",
    # "3": "Ich möchte meine Kontaktinformationen hinterlassen. Zeigen Sie mir das LeadFormular, damit ich es mit meinen Kontaktinformationen ausfüllen kann."
    "1": "Ich möchte wissen, welche Büroflächen derzeit verfügbar sind?",
    "2": "Ich möchte wissen, was den Mietern zur Verfügung steht?",
    "3": "Ich möchte wissen, ob es möglich ist, die Büroräume individuell zu gestalten?",
    "4": "Ich möchte meine Kontaktinformationen hinterlassen. Zeigen Sie mir das Lead-Formular, damit ich es mit meinen Kontaktinformationen ausfüllen kann."
}
############################################################

def default_response_eng_tool_func(_):
    return random.choice(default_replies_eng)

def default_response_deu_tool_func(_):
    return random.choice(default_replies_deu)

def lead_form_tool_func(_):
    return ""

def send_email_to_client(name, email, phone):
    try:
        sg = SendGridAPIClient(api_key=os.getenv('SENDGRID_API_KEY'))
        message = Mail(
            from_email='Contactformblubaar@gmail.com',
            to_emails='olivia.feller@sps.swiss',
            subject='Blu Baar - Ein neues Kontaktformular wurde übermittelt!',
            html_content=f"""
            <h2>Kontaktdaten:</h2>
            <p><strong>Name:</strong> {name}</p>
            <p><strong>Email:</strong> {email}</p>
            <p><strong>Phone:</strong> {phone}</p>
            """
        )

        # Add BCC email address
        personalization = Personalization()
        personalization.add_to(Email('olivia.feller@sps.swiss'))
        personalization.add_bcc(Email('contact-form-blu-baar@navai.ch'))  # Add BCC
        message.add_personalization(personalization)


        response = sg.send(message)
        logging.info(f"Email sent successfully. Status Code: {response.status_code}")
    except Exception as e:
        logging.error(f"Error sending email: {e}")

# Create retriever tools for the glossary
retriever_tool_glossary_eng = create_retriever_tool(
    retriever_glossary_eng,
    "GlossaryRetrieverENG",
    description="Retrieves definitions from the real estate industry glossary. Use only if the user asks for definitions in English."
)

retriever_tool_glossary_deu = create_retriever_tool(
    retriever_glossary_deu,
    "GlossaryRetrieverDEU",
    description="Retrieves definitions from the real estate industry glossary. Use only if the user asks for definitions in German."
)

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
    ensemble_retriever_eng,
    "RetrieverENG",
    description="Retrieves information from the knowledge base. Use only if the user asks questions in English."
)

retriever_tool_deu = create_retriever_tool(
    ensemble_retriever_deu,
    "RetrieverDEU",
    description="Retrieves information from the knowledge base. Use only if the user asks questions in German."
)

pdf_retriever_tool_eng = Tool(
    name="PDFRetrieverENG",
    func=pdf_retriever_eng_tool_func,
    description="Retrieves PDF descriptions and links based on user queries in English. Use this structure: relevant clickable word in square brackets followed by a link in round brackets",
    return_direct=True 
)

pdf_retriever_tool_deu = Tool(
    name="PDFRetrieverDEU",
    func=pdf_retriever_deu_tool_func,
    description="Ruft PDF-Beschreibungen und Links basierend auf Benutzeranfragen in deutscher Sprache ab. Verwenden Sie diese Struktur: relevantes anklickbares Wort in eckigen Klammern, gefolgt von einem Link in runden Klammern",
    return_direct=True 
)

tools = [retriever_tool_eng, 
        retriever_tool_deu, 
        pdf_retriever_tool_eng,  
        pdf_retriever_tool_deu,
        default_response_eng_tool, 
        default_response_deu_tool, 
        lead_form_tool,
        retriever_tool_glossary_eng,
        retriever_tool_glossary_deu]

llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)
system_prompt = read_file('knowledge/system_prompt.txt')

prompt = PromptTemplate(
        template=system_prompt,
        input_variables=[   "input", 
                            # "context", ## no need, because we have retriever_tool_eng and retriever_tool_deu, which model uses.
                            "chat_history"]
    )
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

############################################################
############################################################


@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)


# Flask routes
@socketio.on('chat_message')
def chat(data):

    # RECEIVE data from the route: user_message & user_id & InitialPrompt
    user_message = data.get("message")                  ### 1
    user_id = data.get("user_id")                       ### 2
    initial_prompt = data.get("InitialPrompt", "0")     ### 3

    language = detect_language(user_message)
    message_language[user_id] = language

    # If user_id is not provided, generate a new one
    if not user_id:
        user_id = str(uuid.uuid4())

    # Initialize interaction count and form trigger status if this is a new user
    if user_id not in user_interaction_count:
        user_interaction_count[user_id] = 0  # Start with 0 interactions

    if user_id not in user_form_trigger_status:
        user_form_trigger_status[user_id] = False  # Form has not been shown for this user

    # Increment the interaction count for this user
    user_interaction_count[user_id] += 1

    show_lead_form = False

    if initial_prompt != "0":
        # Use the corresponding initial prompt message
        user_message = initial_prompts.get(initial_prompt, "")
        if not user_message:
            emit('error', {"error": "Invalid InitialPrompt value provided."})
            return
        
        if initial_prompt == "4":
            show_lead_form = True # Since the user wants to leave contact information, we need to trigger the lead form
    else:
        if not user_message:
            emit('error', {"error": "No message provided"})
            return
    

    result = agent_with_chat_history.invoke({"input": user_message}, config={"configurable": {"session_id": user_id}})
    response = result["output"]
    response_content = remove_double_asterisks(response)
    response_content = replace_sharp_s(response_content)

    #FOR DEBUGGING AND TESTING 

    # logger.info(f"User ID: {user_id} - Received message: '{user_message}'")
    log_chat_history(user_id)
    intermediate_steps = result["intermediate_steps"]
    tools_used = [step[0].tool for step in intermediate_steps if step[0].tool]
    # logger.info(f"User ID: {user_id} - Tools used: {tools_used}")
    # logger.info(f"User ID: {user_id} - Assistant response: '{response}'")
    print("user_id: ",user_id)
    print("store:",store) 


    # Check if LeadForm tool was used or interaction count reached 7
    if not show_lead_form:
        if ('LeadForm' in tools_used or user_interaction_count[user_id] == 7): # and not user_form_trigger_status.get(user_id, False):
            # user_form_trigger_status[user_id] = True  # Mark form as shown automatically for this user
            show_lead_form = True
    print("show_lead_form: ", show_lead_form)
    emit('chat_response', {"response": response_content, "user_id": user_id, "show_lead_form": show_lead_form})



@socketio.on('trigger_lead_form')
def trigger_lead_form(data):
    user_id = data.get("user_id")
    name = data.get("name", None)
    email = data.get("email", None)
    phone = data.get("phone", None)

    if not user_id:
        emit('error', {"message": "No user_id provided."})
        return

    # Log the form data, even if some fields are missing
    log_message = f"Lead form submitted by User ID: {user_id}"
    if name:
        log_message += f", Name: {name}"
    if email:
        log_message += f", Email: {email}"
    if phone:
        log_message += f", Phone: {phone}"
    # logger.info(log_message)

    # Send email to the client
    send_email_to_client(name, email, phone)

    # # This code resets status, may be we will need it in the future (not sure)
    # user_form_trigger_status[user_id] = False

    language = message_language.get(user_id, 'DEU')  # Default to 'DEU' if language not set

    if language == 'DEU':
        emit('lead_form_response', {"message": "Vielen Dank für die Kontaktaufnahme. Wir werden uns so rasch wie möglich bei Ihnen melden."})
    else:
        emit('lead_form_response', {"message": "Thank you for contacting us. We will get back to you as soon as possible."}) 




@socketio.on('form_trigger_status')
def form_trigger_status(data):
    user_id = data.get('user_id')
    if not user_id:
        emit('error', {"message": "No user_id provided."})
        return
    
    status = user_form_trigger_status.get(user_id, False)
    # return jsonify({"user_id": user_id, "form_triggered": status})
    emit('form_trigger_status_response', {"user_id": user_id, "form_triggered": status})

############################################################
############################################################
if __name__ == "__main__":
    # eventlet.monkey_patch()
    # app.run(host='0.0.0.0', port=8080, debug=True)
    # app.run(debug=True)
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)
