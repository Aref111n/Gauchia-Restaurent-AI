conversations = {}

def get_or_create_conversation(conversation_id: str):
    if conversation_id not in conversations:
        conversations[conversation_id] = {"messages": [], "active": True}
    return conversations[conversation_id]