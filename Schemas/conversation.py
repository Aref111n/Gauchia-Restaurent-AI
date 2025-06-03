from typing import List, Dict

class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str, str]] = [
            {"role": "system", "content": "You are a helpful restaurent assistent."}
        ]
        self.active: bool = True