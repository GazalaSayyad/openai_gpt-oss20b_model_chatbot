

# 🧠From OpenAI to Open-Source: GPT-oss with Chainlit and LangGraph for Adaptive Agent Design

This project demonstrates how to build a **tool-augmented LLM agent** using `GPT-oss`, `Chainlit`, and `LangGraph`. It supports dynamic tool invocation, persona-based responses, and dual-mode execution—all streamed in real time.

-----

## 🚀 Architecture Overview

This chatbot intelligently switches between two modes based on the user's query:

  * **LangGraph Agent**: Activated by keywords like "weather," "nyc," or "sf," this mode uses tools to gather information and crafts a final response in a specific persona, such as **Al Roker**.
  * **Basic Chatbot**: For all other general queries, it acts as a straightforward assistant, providing streamed responses from a large language model.

This architecture showcases a powerful approach to building LLM applications by combining a robust framework for complex tasks with an efficient fallback for simpler ones.

-----

## 🛠️ Setup

### 1\. Configure Environment Variables

Create a `.env` file in the project's root directory to securely store your API credentials.

```bash
API_KEY=your_fireworks_or_openai_api_key
BASE_URL=https://api.fireworks.ai/inference/v1
```

  * **`API_KEY`**: Your access key for the LLM provider.
  * **`BASE_URL`**: The API endpoint for the model.

### 2\. Install Python Dependencies

Install all required libraries using `pip`.

```bash
pip install -r requirements.txt
```

This command installs essential libraries like LangChain, LangGraph, Chainlit, and the OpenAI SDK.

### 3\. Run the App Locally

Start the chatbot using Chainlit.

```bash
chainlit run app.py
```

This command launches a local web interface at **`http://localhost:8000`**. You can now test the bot with prompts like **"What's the weather in NYC?"** or **"Tell me a joke."**

### 4\. Run with Docker (Optional)

For containerized deployment, use the provided `Dockerfile`.

**Build the Docker Image**

```bash
docker build -t chatbot .
```

This creates a Docker image named `chatbot`.

**Run the Container**

```bash
docker run -p 8000:8000 --env-file .env chatbot
```

  * The `-p 8000:8000` flag maps the container's port to your machine.
  * The `--env-file .env` flag securely passes your API credentials into the container.

Visit **`http://localhost:8000`** to chat with the bot inside Docker.