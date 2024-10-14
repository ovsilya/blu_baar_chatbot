from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import NLTKTextSplitter
from langchain.chains import ConversationalRetrievalChain, create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType, create_react_agent, ZeroShotAgent
from langchain import LLMChain
from nltk.tokenize import sent_tokenize
import os
import os
from flask import Flask, request, jsonify
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
    with open('prompt.txt', 'r') as file:
        return file.read()
    

def load_default_replies():
    with open('default_replies.txt', 'r') as file:
        replies = [line.strip() for line in file if line.strip()]
    return replies


def retrieve_tool_func(query):
    docs = vector_store.similarity_search(query)
    if not docs:
        return ""
    return "\n".join([doc.page_content for doc in docs])


def default_response_tool_func(_):
    return random.choice(default_replies)



retrieve_tool = Tool(
    name="Retriever",
    func=retrieve_tool_func,
    description="Use this tool to retrieve information about Blu-Baar. Always invoke this when the user asks for information about the company or its services."
)


default_response_tool = Tool(
    name="DefaultResponder",
    func=default_response_tool_func,
    description="Use this tool when you don't have enough information to answer the user's question."
)

tools = [retrieve_tool, default_response_tool]



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

# prompt_text = load_prompt_template()

# prompt = ZeroShotAgent.create_prompt(
#     tools=tools,
#     prefix=prompt_template,
#     suffix="",
#     input_variables=["input", "chat_history", "agent_scratchpad"],
# )

# prompt = PromptTemplate(
#     template=prompt_text,
#     input_variables = ["input", "chat_history", "agent_scratchpad", "tool_descriptions"]
# )

def create_langchain_agent(memory):
    prompt_text = load_prompt_template()
    
    prompt = PromptTemplate(
        template = prompt_text,
        input_variables=["input", "context", "chat_history"]
    )
    # print("Tools", prompt.tools)
    # llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Initialize the ZeroShotAgent with the chain and tools
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    # agent = LLMChain(llm=llm, tools=tools, prompt=prompt)
    
    # Create the AgentExecutor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True
    )
    return agent_executor



# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     input_key = "question",
#     return_messages=True,
#     output_key="answer" 
# )


# qa = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever = vector_store.as_retriever(),
#     memory=memory,
#     verbose=True,
#     combine_docs_chain_kwargs={'prompt': PROMPT},
#     return_source_documents=True,
# )


conversation_history = {}

def interact_with_user(user_message, user_id):
    if user_id not in conversation_history:
        # Each user gets their own memory
        conversation_history[user_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input"
        )
    
    memory = conversation_history[user_id]
    agent_executor = create_langchain_agent(memory)

    docs = vector_store.similarity_search(user_message)
    
    context = "\n".join([doc.page_content for doc in docs])
    # response = agent_executor.run(input=user_message)
    # response = agent_executor.run(input=user_message, context=context)
    result = agent_executor.invoke({"input": user_message, "context": context})
    response = result["output"]
    
    conversation_history[user_id] = memory
    
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
