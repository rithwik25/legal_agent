from typing import List, TypedDict

class Message(TypedDict):
    role: str  # "human" or "ai"
    content: str
    timestamp: str

class ChatState(TypedDict):
    """State for the legal chatbot."""
    query: str  # Original user query
    enhanced_query: str  # Query after processing by supervisor
    context: List[str]  # Retrieved legal content
    summary: str  # Simplified legal information
    answer: str  # Final response to user
    legal_references: List[str]  # Sources of information
    conversation_history: List[Message]  # Full conversation history
    next_agent: str  # To control flow in the graph