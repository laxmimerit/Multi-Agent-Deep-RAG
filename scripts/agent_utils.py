"""
Utility functions for agent operations.
"""
from langchain.messages import HumanMessage, AIMessage, ToolMessage


def stream_agent_response(agent, query, thread_id="default"):

    config = {'configurable': {'thread_id': thread_id}}
    
    for chunk in agent.stream(
        {'messages': [HumanMessage(query)]},
        stream_mode='messages',
        config=config
    ):
        # Extract message from chunk
        message = chunk[0] if isinstance(chunk, tuple) else chunk
        
        # Handle AI messages with tool calls
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"\n  Tool Called: {tool_call['name']}")
                print(f"   Args: {tool_call['args']}")
                print()
        
        # Handle tool responses
        elif isinstance(message, ToolMessage):
            print(f"\n  Tool Result (length: {len(message.text)} chars)")
            print()
        
        # Handle AI text responses
        elif isinstance(message, AIMessage) and message.text:
            # Stream the text content
            print(message.text, end='', flush=True)