"""
JSON Parser for extracting valid JSON from LLM responses.
"""
import json
import re
from typing import Optional, Dict, Any


class JSONParser:
    """Robust JSON parser that handles nested structures and code blocks."""
    
    @staticmethod
    def extract(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text using bracket balancing.
        
        Args:
            text: Text containing JSON.
            
        Returns:
            Parsed JSON dict or None if invalid.
        """
        # Try to find JSON in code blocks first
        code_block_pattern = r'```(?:json)?\s*\n(.*?)\n\s*```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        
        for match in matches:
            result = JSONParser._parse_json(match)
            if result:
                return result
        
        # Try to find JSON by bracket balancing
        return JSONParser._find_by_brackets(text)
    
    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Attempt to parse JSON text."""
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None
    
    @staticmethod
    def _find_by_brackets(text: str) -> Optional[Dict[str, Any]]:
        """Find JSON by balancing curly braces."""
        for i, char in enumerate(text):
            if char == '{':
                balance = 1
                j = i + 1
                in_string = False
                escape_next = False
                string_char = None
                
                while j < len(text) and balance > 0:
                    current = text[j]
                    
                    if not escape_next:
                        if current == '\\' and in_string:
                            escape_next = True
                        elif current in ['"', "'"] and not in_string:
                            in_string = True
                            string_char = current
                        elif current == string_char and in_string:
                            in_string = False
                            string_char = None
                        elif not in_string:
                            if current == '{':
                                balance += 1
                            elif current == '}':
                                balance -= 1
                    else:
                        escape_next = False
                    
                    j += 1
                
                if balance == 0:
                    json_str = text[i:j]
                    result = JSONParser._parse_json(json_str)
                    if result:
                        return result
        
        return None
