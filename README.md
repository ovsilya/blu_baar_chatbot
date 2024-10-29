# Custom Chatbot

This repository contains the code for a custom chatbot API built with Python and Flask. It leverages OpenAI's language model API (with LangChain integrations for retrieval and agent handling) and serves responses in both English and German. The application is set up to run on Google Cloud Platform (GCP), utilizing Cloud Build for CI/CD and Cloud Run for containerized deployment.

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

The chatbot is designed to handle interactions in two languages (English and German) and respond with predefined replies when appropriate. It leverages a knowledge base and question-answer documents, which are stored as Chroma vectors for efficient retrieval. It also includes a lead form trigger, which prompts users to enter contact information under certain conditions.

## Features

- **Multi-Language Support:** Supports both English and German language queries.
- **Knowledge Base Retrieval:** Uses Chroma and BM25-based vector storage to retrieve responses from predefined knowledge bases.
- **Lead Form Triggering:** Automatically shows a lead form for collecting user contact information.
- **Interaction History:** Maintains chat history for each user session, allowing for context-aware responses.
- **Dockerized Deployment:** Deploys to GCP with Docker, Cloud Build, and Cloud Run.
- **Logging:** Tracks interactions and stores chat history to facilitate debugging and analytics.

## Project Structure

- `app.py`: Main Flask application that serves as the chatbot API.
- `Dockerfile`: Defines the container setup for deployment on Cloud Run.
- `cloudbuild.yml`: Configuration file for Cloud Build to automate the CI/CD pipeline.
- `knowledge/`: Directory containing English and German knowledge bases (`knowledge_base_eng.txt`, `questions_answers_eng.txt`, etc.).
- `chatbot_interactions.log`: Log file where interaction logs are stored.

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

3. **Set up the environment variables** (see the [Environment Variables](#environment-variables) section).

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

You can load this environment file using `dotenv` or other Python libraries as needed.

## Deployment

This application is deployed on Google Cloud Platform using Cloud Build and Cloud Run.

### Docker Build

To build the Docker image locally:

```bash
docker build -t custom-chatbot .
```

### Deploying with Cloud Build and Cloud Run

#### Step 1: Set up Cloud Build and Cloud Run on GCP

1. Enable Cloud Build and Cloud Run services on GCP.
2. Ensure that you have the necessary permissions for deployment.

#### Step 2: Configure Cloud Build

The `cloudbuild.yml` file contains the configuration for automated deployment using Cloud Build. This configuration triggers on every push to the main branch and deploys the updated container to Cloud Run.

#### Step 3: Deploy the Application

To deploy the app manually using the `gcloud` CLI:

```bash
gcloud builds submit --tag gcr.io/your-project-id/custom-chatbot
gcloud run deploy custom-chatbot --image gcr.io/your-project-id/custom-chatbot --platform managed --region us-central1
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

Other routes include `/trigger-lead-form` for manually submitting lead form data and `/form-trigger-status` for checking the form trigger status.

## Logging

Chat interactions and history are logged in `chatbot_interactions.log`, with logs stored in a rotating format (5 backups, 5MB per file). Logs help with debugging and tracking chatbot interactions.

## Contributing

Please feel free to open issues and submit pull requests to contribute to this project.

## License

This project is licensed under the MIT License.

---

This README provides a comprehensive overview, from project setup to deployment instructions, making it easy for other developers to understand and use your chatbot project. Adjust details like repository links or Google Cloud project names as needed!