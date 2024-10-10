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

app = Flask(__name__)
CORS(app)

main_url = 'https://navai.ch'
second_url = 'https://learnwith.navai.ch'

visited_urls = set()
NAVAICGKEY = os.environ.get('NAVAICGKEY')

# Define the path for the log file
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chatbot_interactions.log')
logging.basicConfig(filename=log_file_path, level=logging.INFO)

leads_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'leads.json')

def fetch_with_retries(url, retries=3, backoff_in_seconds=1):
    for i in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response
        except RequestException as e:
            if i < retries - 1:
                time.sleep(backoff_in_seconds * (2 ** i))
            else:
                raise e

def scrape_site(url):
    if url in visited_urls:
        return []
    visited_urls.add(url)

    response = fetch_with_retries(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    text_content = []
    for element in soup.find_all(string=True):
        if element.parent.name not in ['script', 'style', 'head', 'title', 'meta', '[document]']:
            text_content.append(element.strip())

    links = soup.find_all('a')
    for link in links:
        href = link.get('href')
        if href and not href.startswith(('#', 'javascript:', 'mailto:', '//')) and href.startswith('/'):
            full_url = urljoin(url, href)
            if full_url not in visited_urls:
                text_content.extend(scrape_site(full_url))

    return text_content

all_text_content = scrape_site(main_url)
all_text_content_learn_with_navAI = scrape_site(second_url)

text_content = list(dict.fromkeys(all_text_content))
text_content_program = list(dict.fromkeys(all_text_content_learn_with_navAI))

def load_prompt_template():
    with open('prompt.txt', 'r') as file:
        return file.read()

prompt_template = load_prompt_template()

client = OpenAI(
    api_key=NAVAICGKEY,
    base_url="https://api.openai.com/v1/"
)

def generate_response(prompt_input, text_content, text_content_program):
    string_dialogue = [{"role": "system",
                        "content": prompt_template.format(text_content=text_content, text_content_program=text_content_program)}]
    
    if isinstance(prompt_input, list):
        prompt_input = json.dumps(prompt_input)

    messages = json.loads(prompt_input)
    for dict_message in messages:
        string_dialogue.append(dict_message)

    
    output = client.chat.completions.create(
        model = "gpt-4o",
        messages=string_dialogue
    )

    
    response_text = output.choices[0].message.content
    response_text = append_link_if_needed(messages, response_text)
    formatted_response = format_response(response_text)
    return formatted_response

def append_link_if_needed(messages, response_text):
    last_message = messages[-1]['content'].lower()
    links = {
        "blog": "https://navai.ch/blog/",
        "team": "https://navai.ch/team/",
        "academy": "https://academy.navai.ch",
        "online program": "https://learnwith.navai.ch",
        "learn-with-navAI": "https://learnwith.navai.ch",
        "learn with navai": "https://learnwith.navai.ch"
    }
    
    for keyword, url in links.items():
        if keyword in last_message:
            response_text += f"\nFor more information, please visit {url}."
            break

    return response_text

def remove_double_asterisks(text): 

    pattern = r'\*\*(.*?)\*\*'
    return re.sub(pattern, r'\1', text)

def format_response(response_text):
    lines = response_text.split('\n')
    formatted_lines = []
    for line in lines:
        if line.strip().startswith('-'):
            formatted_lines.append(f"• {line.strip()[1:].strip()}")
        elif line.strip().isdigit():
            formatted_lines.append(f"{line.strip()}.")
        else:
            formatted_lines.append(line.strip())
    return '\n'.join(formatted_lines)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('messages')
        session_id = request.json.get('session_id', str(uuid.uuid4()))

        # Log user interaction with session ID
        logging.info(f"Session ID: {session_id}, User input: {user_input}")

        if isinstance(user_input, list):
            user_input = json.dumps(user_input)
        
        response = generate_response(user_input, text_content, text_content_program)
        # print(response)
        response_content = remove_double_asterisks(response)

        # Log bot response with session ID
        logging.info(f"Session ID: {session_id}, Bot response: {response_content}")
        return jsonify({"content": response_content, "session_id": session_id})
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
    

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/api', methods=['GET'])
def api_info():
    return jsonify({"status": "API is working"}), 200

@app.route('/save_lead', methods=['POST'])
def save_lead():
    try:
        logging.info(f"Request data: {request.data}")
        lead_data = request.json
        logging.info(f"Lead data: {lead_data}")
        
        session_id = lead_data.get('session_id', str(uuid.uuid4()))
        if not lead_data.get('name') or not lead_data.get('email'):
            return jsonify({"error": "Name and Email are required"}), 400

        # Load existing leads
        if os.path.exists(leads_file_path):
            with open(leads_file_path, 'r') as file:
                leads = json.load(file)
        else:
            leads = []

        # Add the new lead
        leads.append(lead_data)

        # Save leads back to the file
        with open(leads_file_path, 'w') as file:
            json.dump(leads, file, indent=4)

        logging.info(f"Session ID: {session_id}, Lead saved: {lead_data}")
        return jsonify({"status": "Lead saved successfully"})
    
    except Exception as e:
        logging.error(f"Error saving lead: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
