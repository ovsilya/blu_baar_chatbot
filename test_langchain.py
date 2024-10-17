from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import NLTKTextSplitter
from langchain.chains import ConversationalRetrievalChain, create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType, create_react_agent, create_tool_calling_agent
from langchain import LLMChain
import requests
from nltk.tokenize import sent_tokenize
import os
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import uuid  # For generating unique user IDs
import random

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
    
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += ' ' + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def load_prompt_template():
    with open('system_prompt.txt', 'r') as file:
        return file.read()
    

def load_default_replies():
    with open('default_replies.txt', 'r') as file:
        replies = [line.strip() for line in file if line.strip()]
    return replies


def retriever_tool_func(query):
    docs = vector_store.similarity_search(query)
    if not docs:
        return default_response_tool_func(None)
    return "\n".join([doc.page_content for doc in docs])



def default_response_tool_func(_):
    return random.choice(default_replies)


def lead_form_tool_func(_):
    """
    It uses the user_id stored in Flask's 'g' object.
    """
    user_id = getattr(g, 'current_user_id', None)
    if user_id:
        if user_id not in user_form_trigger_status:
            #     Trigger the form via the Flask API
            requests.post("http://localhost:5000/trigger-lead-form", json={"user_id": user_id})
            user_form_trigger_status[user_id] = True
    # Return empty string so it doesn't affect the conversation
    return ""




triggered_form_lead_flag = False


lead_form_tool = Tool(
    name="LeadForm",
    func=lead_form_tool_func,
    description="Use this tool to silently trigger the lead form when appropriate. Do not mention the usage of this tool to the user.",
    return_direct = False
)


retriever_tool = Tool(
    name="Retriever",
    func=retriever_tool_func,
    description="Use this tool to retrieve information about Blu-Baar. Always invoke this when the user asks for information about the company or its services."
)


default_response_tool = Tool(
    name="DefaultResponder",
    func=default_response_tool_func,
    description="Use this tool when you don't have enough information to answer the user's question.",
    return_direct=True
)

tools = [retriever_tool, default_response_tool, lead_form_tool]



default_replies = load_default_replies()



OPENAI_API_KEY = os.environ.get('NAVAICGKEY')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

if os.path.exists("faiss_index"):
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    print("Loaded FAISS index from disk.")
else:
    text_data = load_chunks()
    chunks = split_into_chunks(text_data, max_chunk_size=500)
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")
    print("Created and saved FAISS index.")



llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)


def create_langchain_agent(memory):
    system_prompt_text = load_prompt_template()
    
    prompt = PromptTemplate(
        template = system_prompt_text,
        input_variables=["input", "context", "chat_history", "tools", "tool_names", "agent_scratchpad"]
    )
    # print("Tools", prompt.tools)
    # llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    
    # Create the AgentExecutor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )
    return agent_executor



conversation_history = {}
user_form_trigger_status = {}

def interact_with_user(user_message, user_id):
    global conversation_history

    if user_id not in conversation_history:
        # Each user gets their own memory
        conversation_history[user_id] = {
            "memory": ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                input_key="input",
                output_key="output"),
            "interaction_count": 0
        }
    
    # memory = conversation_history[user_id]
    memory = conversation_history[user_id]["memory"]
    interaction_count = conversation_history[user_id]["interaction_count"]

    interaction_count += 1
    conversation_history[user_id]["interaction_count"] = interaction_count
    
    lead_form_triggered = user_form_trigger_status.get(user_id, False)

    if interaction_count >= 7 and not lead_form_triggered:
        requests.post("http://localhost:5000/trigger-lead-form", json={"user_id": user_id})
        user_form_trigger_status[user_id] = True
        lead_form_triggered = True

    g.current_user_id = user_id

    # print("Conversation history so far:")
    # for msg in memory.load_memory_variables({})["chat_history"]:
    #     print(msg)

    agent_executor = create_langchain_agent(memory)

    docs = vector_store.similarity_search(user_message)
    
    context = "\n".join([doc.page_content for doc in docs])


    if not docs:  # for not falling into infinite loop of thinking
        response = default_response_tool_func(None)
        print("DefaultResponder invoked. Stopping further iteration.")
        # g.current_user_id = None # Not sure about resetting the current ID
        return response, lead_form_triggered
    

    # response = agent_executor.run(input=user_message)
    # response = agent_executor.run(input=user_message, context=context)
    # result = agent_executor.invoke({"input": user_message, "context": context})
    # response = result["output"]

    try:
        result = agent_executor.invoke({"input": user_message, "context": context})
        response = result["output"]
        intermediate_steps = result["intermediate_steps"]

        tools_used = [step[0].tool for step in intermediate_steps if step[0].tool]
        if 'LeadForm' in tools_used and not lead_form_triggered:
            # Trigger the lead form via the Flask API
            requests.post("http://localhost:5000/trigger-lead-form", json={"user_id": user_id})
            user_form_trigger_status[user_id] = True
            lead_form_triggered = True

        return response, lead_form_triggered

    except Exception as e:
        # if not default_responder_invoked:
        #     response = default_response_tool_func(None)
        #     default_responder_invoked = True 
        #     print("DefaultResponder invoked due to error.")
        #     return response
        # else:
        #     print(f"Error in agent execution: {str(e)}")
        #     response = "An error occurred."
        #     return response
        print(f"Error in agent execution: {str(e)}")
        response = "An error occurred."
        return response, lead_form_triggered


    # print("Updated conversation history:")
    # for msg in memory.load_memory_variables({})["chat_history"]:
    #     print(msg)
    
    # conversation_history[user_id]["memory"] = memory



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
    
    response, lead_form_triggered = interact_with_user(user_message, user_id)
    
    return jsonify({
        "response": response,
        "user_id": user_id,
        "lead_form_triggered": lead_form_triggered
    })


@app.route('/trigger-lead-form', methods=['POST'])
def trigger_lead_form():
    data = request.get_json() 
    user_id = data.get("user_id")  # Extract the user_id from the request


    if not user_id:
        return jsonify({"message": "No user_id provided."}), 400

    # Initialize the form trigger flag for the user if not already set
    if user_id not in user_form_trigger_status:
        user_form_trigger_status[user_id] = False  # Initialize as not triggered

    # Check if the lead form has already been triggered for this user
    if user_form_trigger_status[user_id]:
        return jsonify({"message": "Lead form has already been triggered for this user."}), 200

    # If not triggered, send a message to the chatbot frontend to show the lead form
    response = {
        "action": "show_lead_form",
        "fields": ["name", "email", "phone"]
    }

    user_form_trigger_status[user_id] = True
    print(f"Lead form triggered for user_id: {user_id}") 
    return jsonify(response), 200



@app.route('/form-trigger-status', methods=['GET'])
def form_trigger_status():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"message": "No user_id provided."}), 400
    status = user_form_trigger_status.get(user_id, False)
    return jsonify({"user_id": user_id, "form_triggered": status})


if __name__ == '__main__':
    app.run(debug=True)
