import logging
import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import time
import tiktoken
import re
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import numpy as np

NAVAICGKEY = os.environ.get('NAVAICGKEY')

client = OpenAI(
    api_key=NAVAICGKEY,
    base_url="https://api.openai.com/v1/"
)

def load_chunks():
    with open('chunks_for_embeddings.txt', 'r') as file:
        return file.read()


def split_into_chunks(text, max_chunk_size=500):
    # Tokenize the text into sentences
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


def generate_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            input=chunk,
            model='text-embedding-3-small'
        )
        embedding = response.data[0].embedding
        embeddings.append({'chunk': chunk, 'embedding': embedding})
    return embeddings



def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_relevant_chunks(query, embeddings, top_n=5):
    # Generate embedding for the query
    response = client.embeddings.create(
        input=query,
        model='text-embedding-3-small'
    )
    query_embedding = response.data[0].embedding
    
    # Calculate similarities
    similarities = []
    for item in embeddings:
        similarity = cosine_similarity(query_embedding, item['embedding'])
        similarities.append({'chunk': item['chunk'], 'similarity': similarity})
    
    # Sort by similarity
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    top_chunks = [item['chunk'] for item in similarities[:top_n]]
    return top_chunks



def load_prompt_template():
    with open('prompt.txt', 'r') as file:
        return file.read()
    

def generate_response(prompt_input, embeddings):
    if isinstance(prompt_input, list):
        user_message = prompt_input[-1]['content']
    else:
        # messages = json.loads(prompt_input)
        # user_message = messages[-1]['content']
        user_message = prompt_input
    
    # Retrieve relevant chunks
    relevant_chunks = find_relevant_chunks(user_message, embeddings, top_n=3)
    context = "\n\n".join(relevant_chunks)
    
    # Construct the system prompt with the context
    system_prompt = prompt_template.format(context=context)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    output = client.chat.completions.create(
        model='gpt-4o',  
        messages=messages
    )
    
    response_text = output.choices[0].message.content

    return response_text



text_data = load_chunks()
chunks = split_into_chunks(text_data, max_chunk_size=500)

print('Chunks: ',chunks[:4])


if not os.path.exists('embeddings.npy'):
    embeddings = generate_embeddings(chunks)
    np.save('embeddings.npy', embeddings)
    print("Generated and saved embeddings.")
else:
    embeddings = np.load('embeddings.npy', allow_pickle=True).tolist()
    print("Loaded embeddings from file.")


# print("Embeddings: ", embeddings[0])

prompt_template = load_prompt_template()

# user_input = 'What is Blu-Baar?'
# user_input = 'Was ist Blu-Baar?'
# user_input = 'Hi! How big are the individual office units?'
user_input = 'Hello! Could you please tell me the size of each individual office unit?'
response = generate_response(user_input,embeddings )

print("Response: \n",response)