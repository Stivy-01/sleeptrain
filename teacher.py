"""
TeacherBrain - The Hippocampus Scorer

Uses Gemini to analyze conversations and decide which memories
are worth consolidating into long-term storage.
"""

import json
from typing import Dict, List, Optional
import google.generativeai as genai

from utils import format_conversation


class TeacherBrain:
    """
    The Teacher Brain acts as the hippocampus - scoring memories
    and generating consolidated "dreams" for training.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the Teacher Brain with Gemini API.
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model to use
        """
        if not api_key or api_key == "YOUR_API_KEY":
            raise ValueError(
                "❌ CRITICAL: You must provide a valid Gemini API key. "
                "Get one at https://makersuite.google.com/app/apikey"
            )
        
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"🎓 Teacher Brain ({model_name}) connected successfully.")
    
    def _call_api(self, prompt: str) -> str:
        """Make API call with error handling"""
        try:
            response = self.model.generate_content(prompt)
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            return clean_text
        except Exception as e:
            print(f"❌ API CALL FAILED: {e}")
            return '{"score": 0, "reason": "API Error"}'
    
    def hippocampus_scan(self, chat_logs: List[Dict[str, str]]) -> Dict:
        """
        Score a conversation for memory importance.
        
        Args:
            chat_logs: List of {"role": str, "content": str} messages
        
        Returns:
            Dict with "score" (1-10) and "reason" fields
        """
        conversation_text = format_conversation(chat_logs)
        
        prompt = f"""
        Analyze this conversation. Rate its importance for long-term memory integration from 1-10.
        - 1-3: Small talk, greetings, transient info (Ignore).
        - 4-7: General context, preferences.
        - 8-10: Critical user facts, identity info, or complex corrections (Must Dream).

        Return JSON only: {{"score": int, "reason": "string"}}

        Conversation:
        {conversation_text}
        """
        
        response = self._call_api(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"score": 0, "reason": "JSON Parse Error"}
    
    def generate_cot_dream(self, chat_logs: List[Dict[str, str]]) -> str:
        """
        Generate a Chain-of-Thought "dream" for memory consolidation.
        
        This creates the ideal response that the student model should learn.
        
        Args:
            chat_logs: The conversation to consolidate
        
        Returns:
            The dream content (ideal response for training)
        """
        conversation_text = format_conversation(chat_logs)
        
        prompt = f"""
        You are a memory manager for an AI.
        The user just provided key identity details.

        Write a NATURAL response that answers the question "Who am I and what do you know about me?".

        The response should:
        1. Be written in the first person ("I know that you are...").
        2. Explicitly state the user's name and details.
        3. Explain the implication (e.g., "Since you are a Python Architect, I will focus on...")

        Do not include <thought> tags. Just give the clear, perfect memory response.

        Conversation Context:
        {conversation_text}
        """
        
        return self._call_api(prompt)
    
    def score_multi_head(
        self, 
        chat_logs: List[Dict[str, str]],
        existing_embeddings: Optional[List] = None
    ) -> Dict:
        """
        Multi-head scoring for more nuanced memory selection.
        
        Returns scores for:
        - salience: novelty relative to existing memories
        - utility: likelihood of being useful in future
        - importance: emotional/factual significance
        - privacy_risk: sensitivity of content
        
        Args:
            chat_logs: The conversation to score
            existing_embeddings: Optional embeddings of existing memories for novelty check
        
        Returns:
            Dict with individual head scores and combined score
        """
        conversation_text = format_conversation(chat_logs)
        
        prompt = f"""
        Analyze this conversation across multiple dimensions.
        Rate each dimension 1-10:

        1. SALIENCE: How novel/unique is this information? (vs generic small talk)
        2. UTILITY: How likely is this to be useful in future conversations?
        3. IMPORTANCE: How critical is this info (identity, corrections, key facts)?
        4. PRIVACY_RISK: How sensitive is this content? (PII, health, financial = high)

        Return JSON only:
        {{
            "salience": int,
            "utility": int, 
            "importance": int,
            "privacy_risk": int,
            "combined_score": int,
            "should_dream": boolean,
            "reason": "string"
        }}

        Conversation:
        {conversation_text}
        """
        
        response = self._call_api(prompt)
        try:
            result = json.loads(response)
            # Ensure all expected fields exist
            defaults = {
                "salience": 5,
                "utility": 5,
                "importance": 5,
                "privacy_risk": 3,
                "combined_score": 5,
                "should_dream": False,
                "reason": "Default"
            }
            for key, default in defaults.items():
                if key not in result:
                    result[key] = default
            return result
        except json.JSONDecodeError:
            return {
                "salience": 0, "utility": 0, "importance": 0,
                "privacy_risk": 0, "combined_score": 0,
                "should_dream": False, "reason": "JSON Parse Error"
            }


if __name__ == "__main__":
    # Test with mock API key (won't actually work without real key)
    print("TeacherBrain module loaded successfully.")
    print("To test, create an instance with: teacher = TeacherBrain('YOUR_API_KEY')")

