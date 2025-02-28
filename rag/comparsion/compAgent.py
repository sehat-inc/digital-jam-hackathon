import json
import re
import google.generativeai as genai

class GeminiAgent:
    """
    A class to interact with the Gemini API for comparing contract summaries.
    """

    # Define the prompt template directly in the class
    COMPARISON_TEMPLATE = """
        Dear AI Assistant,
        I need you to act as an expert Contract Analyst specializing in lease agreement comparisons.
        I have two lease summaries that need to be meticulously compared. Please analyze them with extreme attention to detail and 
        output the result as a JSON object.

        Contract 1:
        {summary1}

        Contract 2:
        {summary2}

        Please perform an exhaustive analysis focusing on these specific areas:

        1. FINANCIAL TERMS ANALYSIS:
        - Compare exact monthly/annual payment amounts
        - List all fees (administrative, processing, late fees)
        - Compare security deposits
        - Analyze payment schedules and due dates
        - Compare any penalties for late payments
        - List any hidden costs or additional charges

        2. LEASE DURATION AND RENEWAL:
        - Compare initial lease terms
        - List all renewal options and conditions
        - Compare notice periods required for renewal
        - Analyze automatic renewal clauses
        - Compare lease extension possibilities
        - List any blackout periods or seasonal restrictions

        3. TERMINATION AND EXIT CONDITIONS:
        - Compare early termination penalties
        - List required notice periods
        - Compare conditions for lease breaking
        - Analyze default conditions
        - Compare cure periods
        - List any special termination rights

        4. OBLIGATIONS AND RESPONSIBILITIES:
        - Compare maintenance responsibilities
        - List insurance requirements
        - Compare utility responsibilities
        - Analyze compliance requirements
        - Compare reporting obligations
        - List any special duties or obligations

        5. SPECIAL PROVISIONS:
        - Compare any unique clauses
        - List special rights or privileges
        - Compare any modification rights
        - Analyze dispute resolution methods
        - Compare force majeure clauses
        - List any unusual restrictions or requirements

        6. RISK ASSESSMENT:
        - Identify potential risks in each agreement
        - Compare liability allocations
        - List indemnification requirements
        - Compare warranty provisions
        - Analyze potential legal exposure
        - Compare compliance requirements

        For each category, provide:
        1. Exact differences with specific details.
        2. Which agreement has more favorable terms (Contract 1, Contract 2, or Neutral).
        3. Specific examples where terms differ.
        4. Note any missing information that should be clarified.
        5. Flag any potentially problematic clauses or conditions.

        **IMPORTANT INSTRUCTIONS FOR RESPONSE FORMATTING:**

        Your response MUST be exclusively in valid JSON format.
        Do NOT include any text, explanations, or comments outside of the JSON object.
        Do NOT enclose the JSON object in Markdown code blocks (like `json ... `).
        The output should be **pure JSON only**, and nothing else.

        Adhere strictly to the following JSON schema:

        ```json
        {{
            "FinancialAnalysis": {{
                "differences": [],
                "favorableAgreement": "Contract 1" or "Contract 2" or "Neutral",
                "concernPoints": [],
                "missingInformation": [],
                "recommendations": []
            }},
            "LeaseTerms": {{ ... }},
            "TerminationProvisions": {{ ... }},
            "ObligationsComparison": {{ ... }},
            "SpecialProvisions": {{ ... }},
            "RiskAssessment": {{ ... }},
            "OverallRecommendation": {{
                "summary": "...",
                "agreementRecommendation": "Contract 1" or "Contract 2" or "Neutral",
                "keyTakeaways": []
            }}
        }}
        ```
        CRITICAL: Your response must be a single, valid JSON object without any markdown formatting or additional text.
        Do not include ```json or ``` markers. Return only the JSON object itself.
    """

    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        """
        Initialize the GeminiAgent with an API key.
        
        Args:
            api_key (str): The API key for Gemini API
            model_name (str, optional): The model to use
        """
        # Configure Gemini with the API key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def compare_summaries(self, summary1, summary2):
        """
        Compare two contract summaries and return a structured analysis.
        
        Args:
            summary1 (str): The first contract summary
            summary2 (str): The second contract summary
            
        Returns:
            dict: The comparison analysis as a JSON object
        """
        # Format the template with the summaries
        prompt = self.COMPARISON_TEMPLATE.format(summary1=summary1, summary2=summary2)
        
        try:
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Parse and clean the JSON response
            return self._parse_json_response(response_text)
            
        except Exception as e:
            return {
                "error": f"Gemini API request failed: {str(e)}"
            }
    
    def _parse_json_response(self, response_text):
        """
        Parse and clean the JSON response from Gemini.
        
        Args:
            response_text (str): Raw text response from Gemini
            
        Returns:
            dict: Parsed JSON or error message
        """
        # Remove code block markers if present
        try:
            cleaned_text = re.sub(r'```json\s*|\s*```', '', response_text)
            cleaned_text = cleaned_text.strip()
            
            # Parse the JSON
            return json.loads(cleaned_text)
            
        except json.JSONDecodeError:
            # Try more aggressive cleanup if needed
            try:
                cleaned_text = re.sub(r'[^\x20-\x7E]', '', cleaned_text)  # Remove non-printable chars
                cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)     # Remove extra newlines
                return json.loads(cleaned_text)
                
            except json.JSONDecodeError:
                return {
                    "error": "Failed to parse JSON response",
                    "raw_response": response_text
                }