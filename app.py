from typing import Literal, Annotated
from typing_extensions import TypedDict
import operator # Import the operator module
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState

import chainlit as cl
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI # Import AsyncOpenAI for the basic chatbot mode

# Load environment variables from .env file
load_dotenv()

# Retrieve API key and base URL from environment variables
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

# Check if environment variables are set
if not BASE_URL:
 raise ValueError("BASE_URL environment variable not set.")
if not API_KEY:
 raise ValueError("API_KEY environment variable not set.")

# Initialize the AsyncOpenAI client directly for the basic chatbot mode

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- Settings for the Basic Chatbot ---
# These settings apply when the LangGraph agent is NOT triggered
basic_chat_settings = {
 "model": "openai/gpt-oss-20b:fireworks-ai",
 "top_p": 0.7,
 "temperature": 0.6,
 "stream": True # Enable streaming for the basic chatbot responses
}

# --- LangChain/LangGraph Setup ---

@tool
def get_weather(city: Literal["nyc", "sf"]):
 """Use this to get weather information."""
 if city == "nyc":
  return "It might be cloudy in nyc"
 elif city == "sf":
  return "It's always sunny in sf"
 else:
  raise ValueError("Invalid city. Please choose 'nyc' or 'sf'.")

# Define the tools available to the LangGraph agent
tools = [get_weather]

# Initialize LangChain ChatOpenAI models
# Updated to use the specified model name and top_p from basic_chat_settings,
# and explicitly passing base_url and api_key for robustness.
model = ChatOpenAI(
 model_name=basic_chat_settings["model"],
 temperature=0, # Retained original temperature
 # FIX: Pass top_p directly as a parameter, not in model_kwargs, to resolve warning
 top_p=basic_chat_settings["top_p"],
 base_url=BASE_URL, # Explicitly pass base_url
 api_key=API_KEY # Explicitly pass api_key
)
final_model = ChatOpenAI(
 model_name=basic_chat_settings["model"],
 temperature=0, # Retained original temperature
 # FIX: Pass top_p directly as a parameter, not in model_kwargs, to resolve warning
 top_p=basic_chat_settings["top_p"],
 base_url=BASE_URL, # Explicitly pass base_url
 api_key=API_KEY # Explicitly pass api_key
)


# Bind tools to the initial model so it can decide to use them
model = model.bind_tools(tools)

# Tag the final model for potential filtering of stream events in Chainlit
final_model = final_model.with_config(tags=["final_node"])

# Create a ToolNode to execute the tools defined
tool_node = ToolNode(tools=tools)

# Define the state for the graph, which is a list of messages
class AgentState(TypedDict):
 # FIX: Changed `lambda x: x` to `operator.add` as the reducer for lists.
 # LangGraph expects reducers for list channels to take two arguments (current list, new item(s))
 # and return the updated list. operator.add performs list concatenation.
 messages: Annotated[list[BaseMessage], operator.add]

# Define the conditional logic for routing within the graph
def should_continue(state: AgentState) -> Literal["tools", "final"]:
 """
 Determines whether the agent needs to use tools or can provide a final answer.
 """
 messages = state["messages"]
 last_message = messages[-1]
 # If the LLM's last message includes tool calls, route to the "tools" node
 if last_message.tool_calls:
  return "tools"
 # Otherwise, it means the LLM has a direct answer, so route to the "final" node
 return "final"

# Node to call the primary LLM model
def call_model(state: AgentState):
 """
 Invokes the main LLM to generate a response based on the current conversation history.
 """
 messages = state["messages"]
 response = model.invoke(messages)
 return {"messages": [response]}

# Node to call the final LLM model, typically to rephrase or finalize the answer
def call_final_model(state: AgentState):
 """
 Invokes a separate LLM (or the same one with a different prompt)
 to refine the final answer, e.g., in a specific persona.
 """
 messages = state["messages"]
 last_ai_message = messages[-1] # Get the last AI message from the agent
 # Call the final model with a system message to set the persona (Al Roker)
 response = final_model.invoke(
  [
   SystemMessage("Rewrite this in the voice of Al Roker, a cheerful and enthusiastic meteorologist. Make it sound like a weather report."),
   HumanMessage(last_ai_message.content), # The content to be rephrased
  ]
 )
 # Overwrite the ID of the new response with the original last AI message's ID.
 # This helps Chainlit correctly associate the streamed output with the agent's turn.
 response.id = last_ai_message.id
 return {"messages": [response]}

# Build the LangGraph
builder = StateGraph(AgentState)

# Add nodes to the graph representing different stages of the agent's process
builder.add_node("agent", call_model) # Main LLM decision-making
builder.add_node("tools", tool_node) # Tool execution
builder.add_node("final", call_final_model) # Final response generation/rephrasing

# Define the entry point of the graph
builder.add_edge(START, "agent")

# Define conditional transitions from the "agent" node based on should_continue logic
builder.add_conditional_edges(
 "agent",
 should_continue,
 {"tools": "tools", "final": "final"}, # If tool calls, go to "tools"; otherwise, go to "final"
)

# Define transition after tool execution: always return to the "agent" to re-evaluate
builder.add_edge("tools", "agent")

# Define the exit point for the final response from the "final" node
builder.add_edge("final", END)

# Compile the graph for execution
graph = builder.compile()

# --- Chainlit on_message function with dual mode logic ---

@cl.on_message
async def on_message(msg: cl.Message):
    content = msg.content.lower()

    if any(keyword in content for keyword in ["weather", "nyc", "sf"]):
        # --- LangGraph with Tools Logic ---
        try:
            # A more robust way to run the graph is to use ainvoke and get the final state.
            final_state = await graph.ainvoke({"messages": [HumanMessage(content=msg.content)]})
            
            # The final message is the last one in the state's message list.
            final_message_content = final_state["messages"][-1].content

            # Create a new Chainlit message to stream the final answer.
            final_answer = cl.Message(content="")
            await final_answer.stream_token(final_message_content)
            await final_answer.send()

        except Exception as e:
            print(f"Error processing LangGraph message: {e}")
            await cl.Message(content=f"Oops! Something went wrong with the LangGraph agent. Please try again. Error: {e}").send()

    else:
        # --- Basic Chatbot Logic ---
        try:
            msg_basic = cl.Message(content="")
            await msg_basic.send()

            response = await client.chat.completions.create(
                messages=[
                    {
                        "content": "You are a helpful general-purpose chatbot. You always reply in a polite and conversational manner.",
                        "role": "system"
                    },
                    {
                        "content": msg.content,
                        "role": "user"
                    }
                ],
                **basic_chat_settings
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    await msg_basic.stream_token(chunk.choices[0].delta.content)

            await msg_basic.send()

        except Exception as e:
            print(f"Error processing basic chat message: {e}")
            await cl.Message(content=f"Oops! Something went wrong with the basic chat. Please try again. Error: {e}").send()
            
            # from typing import Literal, Annotated
# from typing_extensions import TypedDict
# import operator # Import the operator module
# from langchain_core.tools import tool
# from langchain_openai import ChatOpenAI
# from langgraph.prebuilt import ToolNode
# from langchain.schema.runnable.config import RunnableConfig
# from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
# from langgraph.graph import END, StateGraph, START
# from langgraph.graph.message import MessagesState

# import chainlit as cl
# from dotenv import load_dotenv
# import os
# from openai import AsyncOpenAI # Import AsyncOpenAI for the basic chatbot mode

# # Load environment variables from .env file
# load_dotenv()

# # Retrieve API key and base URL from environment variables
# BASE_URL = os.getenv("BASE_URL")
# API_KEY = os.getenv("API_KEY")

# # Check if environment variables are set
# if not BASE_URL:
#     raise ValueError("BASE_URL environment variable not set.")
# if not API_KEY:
#     raise ValueError("API_KEY environment variable not set.")

# # Initialize the AsyncOpenAI client directly for the basic chatbot mode
# client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# # --- Settings for the Basic Chatbot ---
# # These settings apply when the LangGraph agent is NOT triggered
# basic_chat_settings = {
#     "model": "openai/gpt-oss-20b:fireworks-ai",
#     "top_p": 0.7,
#     "temperature": 0.6,
#     "stream": True # Enable streaming for the basic chatbot responses
# }

# # --- LangChain/LangGraph Setup ---

# @tool
# def get_weather(city: Literal["nyc", "sf"]):
#     """Use this to get weather information."""
#     if city == "nyc":
#         return "It might be cloudy in nyc"
#     elif city == "sf":
#         return "It's always sunny in sf"
#     else:
#         raise ValueError("Invalid city. Please choose 'nyc' or 'sf'.")

# # Define the tools available to the LangGraph agent
# tools = [get_weather]

# # Initialize LangChain ChatOpenAI models
# # Updated to use the specified model name and top_p from basic_chat_settings,
# # and explicitly passing base_url and api_key for robustness.
# model = ChatOpenAI(
#     model_name=basic_chat_settings["model"],
#     temperature=0, # Retained original temperature
#     # FIX: Pass top_p directly as a parameter, not in model_kwargs, to resolve warning
#     top_p=basic_chat_settings["top_p"],
#     base_url=BASE_URL, # Explicitly pass base_url
#     api_key=API_KEY # Explicitly pass api_key
# )
# final_model = ChatOpenAI(
#     model_name=basic_chat_settings["model"],
#     temperature=0, # Retained original temperature
#     # FIX: Pass top_p directly as a parameter, not in model_kwargs, to resolve warning
#     top_p=basic_chat_settings["top_p"],
#     base_url=BASE_URL, # Explicitly pass base_url
#     api_key=API_KEY # Explicitly pass api_key
# )


# # Bind tools to the initial model so it can decide to use them
# model = model.bind_tools(tools)

# # Tag the final model for potential filtering of stream events in Chainlit
# final_model = final_model.with_config(tags=["final_node"])

# # Create a ToolNode to execute the tools defined
# tool_node = ToolNode(tools=tools)

# # Define the state for the graph, which is a list of messages
# class AgentState(TypedDict):
#     # FIX: Changed `lambda x: x` to `operator.add` as the reducer for lists.
#     # LangGraph expects reducers for list channels to take two arguments (current list, new item(s))
#     # and return the updated list. operator.add performs list concatenation.
#     messages: Annotated[list[BaseMessage], operator.add]

# # Define the conditional logic for routing within the graph
# def should_continue(state: AgentState) -> Literal["tools", "final"]:
#     """
#     Determines whether the agent needs to use tools or can provide a final answer.
#     """
#     messages = state["messages"]
#     last_message = messages[-1]
#     # If the LLM's last message includes tool calls, route to the "tools" node
#     if last_message.tool_calls:
#         return "tools"
#     # Otherwise, it means the LLM has a direct answer, so route to the "final" node
#     return "final"

# # Node to call the primary LLM model
# def call_model(state: AgentState):
#     """
#     Invokes the main LLM to generate a response based on the current conversation history.
#     """
#     messages = state["messages"]
#     response = model.invoke(messages)
#     return {"messages": [response]}

# # Node to call the final LLM model, typically to rephrase or finalize the answer
# def call_final_model(state: AgentState):
#     """
#     Invokes a separate LLM (or the same one with a different prompt)
#     to refine the final answer, e.g., in a specific persona.
#     """
#     messages = state["messages"]
#     last_ai_message = messages[-1] # Get the last AI message from the agent
#     # Call the final model with a system message to set the persona (Al Roker)
#     response = final_model.invoke(
#         [
#             SystemMessage("Rewrite this in the voice of Al Roker, a cheerful and enthusiastic meteorologist. Make it sound like a weather report."),
#             HumanMessage(last_ai_message.content), # The content to be rephrased
#         ]
#     )
#     # Overwrite the ID of the new response with the original last AI message's ID.
#     # This helps Chainlit correctly associate the streamed output with the agent's turn.
#     response.id = last_ai_message.id
#     return {"messages": [response]}

# # Build the LangGraph
# builder = StateGraph(AgentState)

# # Add nodes to the graph representing different stages of the agent's process
# builder.add_node("agent", call_model) # Main LLM decision-making
# builder.add_node("tools", tool_node) # Tool execution
# builder.add_node("final", call_final_model) # Final response generation/rephrasing

# # Define the entry point of the graph
# builder.add_edge(START, "agent")

# # Define conditional transitions from the "agent" node based on should_continue logic
# builder.add_conditional_edges(
#     "agent",
#     should_continue,
#     {"tools": "tools", "final": "final"}, # If tool calls, go to "tools"; otherwise, go to "final"
# )

# # Define transition after tool execution: always return to the "agent" to re-evaluate
# builder.add_edge("tools", "agent")

# # Define the exit point for the final response from the "final" node
# builder.add_edge("final", END)

# # Compile the graph for execution
# graph = builder.compile()

# # --- Chainlit on_message function with dual mode logic ---

# @cl.on_message
# async def on_message(msg: cl.Message):
#     content = msg.content.lower()

#     # Determine whether to use LangGraph (tool-enabled) or the basic chatbot
#     # We'll trigger LangGraph if the message contains specific keywords
#     if any(keyword in content for keyword in ["weather", "nyc", "sf"]):
#         # --- LangGraph with Tools Logic ---
#         try:
#             # Configure Chainlit session for LangGraph thread ID to track conversations
#             # Removed config = {"configurable": {"thread_id": cl.context.session.id}}
#             # As this was likely contributing to the __enter__ error
#             # Create a new Chainlit message to stream the final answer to the user
#             final_answer = cl.Message(content="")

#             # Stream the graph execution. stream_mode="messages" yields the state's messages
#             # after each node execution, allowing for real-time updates.
#             # FIX: Removed 'callbacks=[cb]' from RunnableConfig because cb is no longer used
#             # FIX: Removed 'config=RunnableConfig(**config)' from graph.stream call
#             for s_msg in graph.stream(
#                 {"messages": [HumanMessage(content=msg.content)]},
#                 stream_mode="messages"
#             ):
#                 # LangGraph streams dictionary updates to the state. We're interested in the 'messages' key.
#                 if "messages" in s_msg:
#                     current_message = s_msg["messages"][-1]
#                     # Only stream content from AI messages, and specifically from the "final" node's output
#                     # when the graph is concluding.
#                     if isinstance(current_message, BaseMessage) and not isinstance(current_message, HumanMessage):
#                         # We capture the final message content, ensuring we only stream from the intended final output.
#                         # This checks for the 'final' node in the graph's end signal or if the message is tagged.
#                         if s_msg.get("__end__") == "final" or ("tags" in current_message.response_metadata and "final_node" in current_message.response_metadata["tags"]):
#                             await final_answer.stream_token(current_message.content)

#             await final_answer.send() # Send the complete streamed message to the user

#         except Exception as e:
#             # Catch any exceptions that occur during the process (e.g., network issues, API errors)
#             print(f"Error processing LangGraph message: {e}")
#             # Send a user-friendly error message back to the chat interface
#             await cl.Message(content=f"Oops! Something went wrong with the LangGraph agent. Please try again. Error: {e}").send()

#     else:
#         # --- Basic Chatbot Logic ---
#         try:
#             # Create a message object for streaming the response for the basic chatbot
#             msg_basic = cl.Message(content="")
#             await msg_basic.send() # Send the initial empty message to show loading

#             # Make the API call with streaming enabled using the AsyncOpenAI client
#             response = await client.chat.completions.create(
#                 messages=[
#                     {
#                         "content": "You are a helpful general-purpose chatbot. You always reply in a polite and conversational manner.",
#                         "role": "system"
#                     },
#                     {
#                         "content": msg.content,
#                         "role": "user"
#                     }
#                 ],
#                 **basic_chat_settings # Use the basic chatbot settings
#             )

#             # Iterate over the response chunks received from the streaming API
#             async for chunk in response:
#                 if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
#                     await msg_basic.stream_token(chunk.choices[0].delta.content) # Stream each token

#             await msg_basic.send() # Send the final message once all chunks are received

#         except Exception as e:
#             # Catch any exceptions that occur during the basic chat process
#             print(f"Error processing basic chat message: {e}")
#             await cl.Message(content=f"Oops! Something went wrong with the basic chat. Please try again. Error: {e}").send()
