# Blu Baar Chatbot — Bilingual (EN/DE) Agentic RAG Assistant

A real-time, bilingual (English/German) customer-facing chatbot API built with **Flask-SocketIO** and a **LangChain tool-calling agent** over **GPT-4o**. It answers from a curated knowledge base using **hybrid retrieval (Chroma + BM25 ensemble)** and can trigger a **lead-capture form** that emails contact details via SendGrid. Containerized and deployed on **Google Cloud Run** (Cloud Build CI/CD).

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Retrieval](#retrieval)
- [Agent Tools](#agent-tools)
- [Lead Capture](#lead-capture)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Local Setup](#local-setup)
- [Deployment](#deployment)
- [Socket.IO API](#socketio-api)
- [Logging](#logging)

## Overview

The assistant handles website visitor queries in **English and German**, detecting the language per message and routing to language-specific retrieval tools. Answers are grounded in a knowledge base (general info, Q&A pairs, glossary/terminology, and PDF-document metadata). When appropriate, it surfaces a lead-capture form to collect visitor contact details. Communication is **real-time over WebSockets** (Socket.IO), not request/response REST.

## Features

- **Bilingual (EN/DE)** — per-message language detection (`langdetect`) with language-scoped tools.
- **Agentic RAG** — a LangChain tool-calling agent (GPT-4o) chooses which retriever tool to invoke per query.
- **Hybrid retrieval** — Chroma (semantic) + BM25 (lexical) combined via `EnsembleRetriever`, per language.
- **Lead capture** — multi-condition trigger; submissions emailed via SendGrid.
- **Session memory** — per-session chat history for context-aware, multi-turn conversations.
- **Real-time transport** — Flask-SocketIO (eventlet) WebSocket server.
- **Cloud-native** — Docker + Cloud Build + Cloud Run (EU region).

## Architecture

Per incoming `chat_message` event:
1. Client emits `{ message, user_id, InitialPrompt }` over Socket.IO.
2. **Language detection** (`langdetect`) → EN or DE → selects language-scoped tools.
3. Message + session `chat_history` → **tool-calling agent** (`create_tool_calling_agent` + `AgentExecutor`, GPT-4o) → the agent decides which retriever tool(s) to call.
4. Retrieved context → grounded answer; light normalization (strip markdown `*`, German `ß`→`ss`).
5. Lead-form logic evaluated → `show_lead_form` flag returned with the response.
6. Session history updated.

Knowledge is indexed into **Chroma** vector stores, built per language (EN/DE): **knowledge base**, **Q&A pairs**, **glossary**, and **PDF descriptions** (metadata + links; source CSV pulled from a Google Sheet). Documents are chunked with `RecursiveCharacterTextSplitter` (chunk size 500, overlap 100).

## Retrieval

Hybrid retrieval via LangChain **`EnsembleRetriever`**, combining per language:
- **Chroma** vector retriever (dense / semantic)
- **BM25** retriever (`rank-bm25`, sparse / lexical)
- **Q&A** vector retriever (semantic)

Separate EN and DE ensembles; the PDF-metadata retriever is a standalone Chroma instance. Combining dense and sparse retrieval improves recall on FAQ/KB content where exact-term matches (product names, conditions) matter alongside semantics.

## Agent Tools

Exposed to the agent (each scoped to a language by its description):
- `RetrieverENG` / `RetrieverDEU` — knowledge-base ensemble
- `PDFRetrieverENG` / `PDFRetrieverDEU` — PDF metadata with markdown-link formatting
- `GlossaryRetrieverENG` / `GlossaryRetrieverDEU` — term definitions
- `DefaultResponderENG` / `DefaultResponderDEU` — fallback responses
- `LeadForm` — silent trigger the agent can invoke to surface the lead form

The system prompt is loaded from `knowledge/system_prompt.txt`.

## Lead Capture

The lead form is surfaced (`show_lead_form = true`) when **any** of:
1. the user selects initial-prompt option `"4"` (explicit request), **or**
2. the agent invokes the `LeadForm` tool, **or**
3. the interaction count for the session reaches **7** messages.

On submission (`trigger_lead_form`), an HTML email with the visitor's name/email/phone is sent via **SendGrid**.

## Project Structure

- `app.py` — Flask-SocketIO server: agent, retrievers, tools, memory, lead logic, email.
- `client.py` — reference Socket.IO test client.
- `Dockerfile` — container config (python:3.12, non-root user, port 8080).
- `cloudbuild.yaml` — Cloud Build CI/CD (build → GCR → Cloud Run).
- `knowledge/` — knowledge-base, Q&A, glossary, and system-prompt files (EN/DE).
- `requirements.txt` — dependencies.

## Environment Variables

- **`OPENAI_API_KEY`** — OpenAI API key for GPT-4o (required).
- **`SENDGRID_API_KEY`** — SendGrid API key for lead-form emails (required).
- **`GOOGLE_PATH`** — URL of the Google Sheet (CSV export) holding PDF-document metadata.

## Local Setup

```bash
git clone https://github.com/ovsilya/blu_baar_chatbot.git
cd blu_baar_chatbot
pip install -r requirements.txt
# set env vars (e.g. in a .env file): OPENAI_API_KEY, SENDGRID_API_KEY, GOOGLE_PATH
python app.py
```

The Socket.IO server runs on `http://localhost:8080`.

## Deployment

Deployed on **Google Cloud Run** via **Cloud Build** (`cloudbuild.yaml`):
1. Build the Docker image (`gcr.io/$PROJECT_ID/chatbot-app-websocket`).
2. Push to Google Container Registry.
3. `gcloud run deploy` to Cloud Run (managed), region **`europe-west1`**, `--allow-unauthenticated`.

```bash
gcloud builds submit
```

## Socket.IO API

WebSocket events (Socket.IO):

| Event | Direction | Purpose |
|---|---|---|
| `connect` / `disconnect` | client ↔ server | connection lifecycle |
| `chat_message` | client → server | `{ message, user_id, InitialPrompt }`; primary chat |
| `chat_response` | server → client | `{ response, user_id, show_lead_form }` |
| `trigger_lead_form` | client → server | submit lead form (name/email/phone) → SendGrid email |
| `form_trigger_status` | client → server | query per-user form-display status |
| `user_id` | server → client | assigned session/user id |

See `client.py` for a minimal client example.

## Logging

Chat interactions are logged with a rotating file handler (up to 5 backups of 5 MB each) for debugging and tracking. On Cloud Run, logs are also captured via Cloud Logging.

## Tech Stack

Python · LangChain (`create_tool_calling_agent`, `EnsembleRetriever`, `RunnableWithMessageHistory`) · OpenAI GPT-4o (`langchain-openai`) · Chroma · BM25 (`rank-bm25`) · Flask · Flask-SocketIO · eventlet · SendGrid · `langdetect` · pandas · Docker · GCP Cloud Build + Cloud Run.
