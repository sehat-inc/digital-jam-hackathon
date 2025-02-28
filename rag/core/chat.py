import os
import google.generativeai as genai
from typing import List, Dict, Any
import time

from vector_store import RetrievalChunks


HISTORY_LENGTH = 5 

class RAGChatbot:
    def __init__(self, model, index, api_key: str):
        self.api_key = api_key
        self.model = self._init_model()
        self.conversation_history = []
        self.max_history = HISTORY_LENGTH
        self.retrieve = RetrievalChunks(model, index)

        
    def _init_model(self):
        """Initialize the Gemini model."""
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    
    def retrieve_context(self, query: str) -> List[str]:
        """Retrieve relevant chunks for the query using the existing retrieve_chunks function."""
        return self.retrieve.retrieve_chunks(query)
        
    def format_conversation_history(self) -> str:
        """Format the conversation history for context."""
        if not self.conversation_history:
            return ""
            
        formatted_history = "Previous conversation:\n"
        for i, exchange in enumerate(self.conversation_history):
            formatted_history += f"User: {exchange['user']}\n"
            formatted_history += f"Assistant: {exchange['assistant']}\n"
            
        return formatted_history
    
    def generate_prompt(self, query: str) -> str:
        """Generate the full prompt with context, history, and current query."""
        # Retrieve relevant context
        context_chunks = self.retrieve_context(query)
        
        # Format context chunks
        context_text = "\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Get conversation history
        history_text = self.format_conversation_history()
        
        # Construct the full prompt
        prompt = f"""
You are a helpful legal assistant that answers questions based on the provided context.
If you don't know the answer or if the context doesn't contain relevant information, 
just say so instead of making up information.

{context_text}

{history_text}

Current question: {query}

Answer:
"""
        return prompt
    
    def update_history(self, user_query: str, assistant_response: str):
        """Update the conversation history, maintaining the max history length."""
        self.conversation_history.append({
            "user": user_query,
            "assistant": assistant_response
        })
        
        # Keep only the most recent conversations
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    async def generate_response_stream(self, query: str):
        """Generate a streaming response to the user query."""
        prompt = self.generate_prompt(query)
        
        # Create a streaming response
        response_stream = await self.model.generate_content_async(
            prompt,
            generation_config={"temperature": 0.2},
            stream=True
        )
        
        full_response = ""
        
        # Process and yield each chunk as it arrives
        async for chunk in response_stream:
            if hasattr(chunk, 'text'):
                yield chunk.text
                full_response += chunk.text
        
        # Update conversation history with the complete response
        self.update_history(query, full_response)
        
    def generate_response(self, query: str) -> str:
        """Generate a non-streaming response (for cases where streaming isn't needed)."""
        prompt = self.generate_prompt(query)
        
        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": 0.2}
        )
        
        response_text = response.text
        
        # Update conversation history
        self.update_history(query, response_text)
        
        return response_text
    
#while True:
#    user_input = input("\nYou: ")
#
#    if user_input.lower() == 'exit':
#        print("Goodbye!")
#        break
#    print("\nAssistant: ", end="")
#
#    async for text_chunk in chatbot.generate_response_stream(user_input):
#        print(text_chunk, end="", flush=True)
#        time.sleep(0.01)