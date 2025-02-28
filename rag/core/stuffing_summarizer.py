from typing import Optional, Any
import argparse

import google.generativeai as genai
from dotenv import load_dotenv
import os 

load_dotenv()

class SummarizerAgent:
    def __init__(self,  llm: Any, role: Optional[str] = None, background: Optional[str] = None):
        """
        Initialize an AI Agent with an optional system prompt 

        Args: 
            llm: Language Model
            role: Role of the AI Agent 
            background: Background instructions for the AI agent

        """

        self.llm = llm
        
        self.messages = []

        default_role = "Contract Specialist/Contract Manager" 

        default_background = """ You are an expert Contract Specialist tasked with summarizing 
        legal lease contracts. Follow these rules: 
        1. Simplify legal jargon and complex terms
        2. Make sure to include and highlight key terms 
        3. Summarize obligations if stated 
        4. Include important dates if stated 
        5. Clarify termination clauses if stated 
        6. Explain renewal terms if stated 
        7. Pay attention to penalties and fees 
        8. Include ALL information regardless if it is confidential
        9. Summarize special provisions to ensure understanding 
        10. Organize information logically 
        11. Provide examples where you deem necessary to clarify complex clauses or obligations
        Remember to summarize the documents in mostly bullet points for more readability
        """

        self.prompt_template = f"""
        Role: {role or default_role}

        Background: {background or default_background}

        {{text}}
        
        """
    
    def _run(self, text: str) -> str:
        """
        Process and summarize the input text. 

        Args: 
            text (str): The contract text to be summarized

        Returns: 
            str: Summarized contract with key points and explanations
        """
        try: 
            prompt = self.prompt_template.format(text=text)

            response = self.llm.generate_content(prompt)

            return response.text

        except Exception as e: 
            return f"Error processing contract: {str(e)}"
