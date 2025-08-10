from openai import AsyncOpenAI
import chainlit as cl
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve API key and base URL from environment variables
# Use os.getenv() to get the variable, providing a default or raising an error if not found
# Updated variable names to BASE_URL and API_KEY as per your request
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

# Check if environment variables are set
if not BASE_URL:
    raise ValueError("BASE_URL environment variable not set.")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set.")

# Initialize the OpenAI client with variables from the environment
# Using the updated BASE_URL and API_KEY
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# Instrument the OpenAI client (assuming this is for Chainlit's tracing)
cl.instrument_openai()

settings = {
    "model": "openai/gpt-oss-20b:fireworks-ai",
    "top_p": 0.7,
    "temperature": 0.6,
    "stream": True # This enables streaming from the OpenAI API, crucial for chunking
}

@cl.on_message
async def on_message(message: cl.Message):
    # This try-except block handles potential issues during API calls
    try:
        # Create a message object for streaming the response.
        # Removed 'stream=True' from cl.Message initialization, as it caused an error.
        # Chainlit handles streaming internally when stream_token is used.
        msg = cl.Message(content="")
        await msg.send() # Send the initial empty message to show loading to the user

        # Make the API call with streaming enabled
        response = await client.chat.completions.create(
            messages=[
                {
                    "content": "You are a helpful bot, you always reply in polite with reasoning capability",
                    "role": "system"
                },
                {
                    "content": message.content,
                    "role": "user"
                }
            ],
            **settings
        )

        # Iterate over the response chunks received from the streaming API
        async for chunk in response:
            # Check if there's content in the current chunk
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                # Append each piece of content (token) to the message as it arrives.
                # This provides the real-time "typing" effect in the UI.
                await msg.stream_token(chunk.choices[0].delta.content)

        # After all chunks have been received and streamed, send the final message.
        # This marks the message as complete in the UI.
        await msg.send()

    except Exception as e:
        # Catch any exceptions that occur during the process (e.g., network issues, API errors)
        print(f"Error processing message: {e}")
        # Send a user-friendly error message back to the chat interface
        await cl.Message(content=f"Oops! Something went wrong. Please try again. Error: {e}").send()

