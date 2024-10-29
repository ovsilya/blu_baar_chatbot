# Custom Chatbot

This repository contains the code for a custom chatbot API, built with Python and Flask, that uses OpenAI's language model API (with LangChain integrations) to handle queries in English and German. The app is designed to run on Google Cloud Platform (GCP) using Cloud Build for continuous integration and Cloud Run for containerized deployment.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)
- [Usage](#usage)
- [Logging](#logging)
- [Contributing](#contributing)

## Project Overview

The chatbot processes queries in English and German, retrieving relevant responses from a knowledge base. When appropriate, it prompts users with a lead form for gathering contact details. Responses are enhanced by using both Chroma and BM25 retrievers for better information retrieval.

## Features

- **Multi-Language Support:** Answers queries in both English and German.
- **Knowledge Base Retrieval:** Utilizes Chroma and BM25-based vector storage for quick and accurate responses.
- **Lead Form Triggering:** Automatically prompts users to submit contact information.
- **Persistent Interaction History:** Saves session-based chat history to enable context-aware responses.
- **Dockerized Deployment:** Deployed on GCP with Docker, Cloud Build, and Cloud Run.
- **Logging:** Maintains logs of all chat interactions for debugging and tracking purposes.

## Project Structure

- `app.py`: Main Flask application serving as the chatbot API.
- `Dockerfile`: Specifies the container configuration for Cloud Run.
- `cloudbuild.yml`: Configures Cloud Build to automate CI/CD pipeline for deployment.
- `knowledge/`: Directory containing knowledge base and question-answer files (`knowledge_base_eng.txt`, `questions_answers_eng.txt`, etc.).
- `chatbot_interactions.log`: Log file for chat interactions.

## Setup Instructions

### Prerequisites

- Python 3.8+
- Docker
- Google Cloud SDK (for deployment)

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (see the [Environment Variables](#environment-variables) section).

4. **Run the application locally:**
   ```bash
   python app.py
   ```

   The application will be available at `http://localhost:8080`.

### Environment Variables

The application requires the following environment variables:

- **`OPENAI_API_KEY`**: API key for OpenAI's language model (required).

To set these variables locally, create a `.env` file:

```bash
OPENAI_API_KEY="your_openai_api_key"
```

## Deployment

This application is deployed on Google Cloud Platform (GCP) using Cloud Build and Cloud Run. The process is automated using `cloudbuild.yml`, which handles the build, push, and deployment steps.

### Docker Build

To build the Docker image locally:

```bash
docker build -t custom-chatbot .
```

### Google Cloud Platform Deployment

#### Step 1: Set up Cloud Build and Cloud Run on GCP

1. Enable **Cloud Build** and **Cloud Run** services on GCP.
2. Ensure permissions are granted for deployment.

#### Step 2: Configure Cloud Build

The `cloudbuild.yml` file defines the CI/CD pipeline:
- **Step 1**: Builds the Docker image using `gcr.io/cloud-builders/docker`.
- **Step 2**: Pushes the image to Google Container Registry (GCR).
- **Step 3**: Deploys the image to Cloud Run using `gcloud run deploy`.

#### Step 3: Deployment Command (Manual)

To deploy the app manually with `gcloud`:

```bash
gcloud builds submit --tag gcr.io/your-project-id/chatbot-app
gcloud run deploy chatbot-app --image gcr.io/your-project-id/chatbot-app --platform managed --region us-central1 --allow-unauthenticated
```

## Usage

The API accepts POST requests to the `/chat` endpoint, with the following structure:

- **Endpoint**: `/chat`
- **Method**: `POST`
- **Request Payload**:
  ```json
  {
    "message": "Your query here",
    "user_id": "unique_user_id",
    "InitialPrompt": "optional_initial_prompt_code"
  }
  ```

- **Response Payload**:
  ```json
  {
    "response": "Chatbot response",
    "user_id": "unique_user_id",
    "show_lead_form": true
  }
  ```

Additional routes:
- `/trigger-lead-form` for submitting lead form data.
- `/form-trigger-status` for checking form trigger status.

## Logging

Chat interactions are logged in `chatbot_interactions.log`, with a rotating log setup to save up to 5 backup files of 5MB each. These logs help with debugging and tracking user interactions.