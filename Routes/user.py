from fastapi import HTTPException, APIRouter
from Schemas.userinput import UserInput
from Services.user_service import query_groq_api, get_or_create_conversation, load_dataset

route = APIRouter()

@route.post("/chat/")
async def chat(input: UserInput):
    conversation = get_or_create_conversation(input.conversation_id)

    if not conversation.active:
        raise HTTPException(
            status_code=400,
            detail="The chat session has ended. Please start a new session."
        )

    try:
        conversation.messages.append({
            "role": input.role,
            "content": input.message
        })

        dataset = load_dataset()  

        if isinstance(dataset, list):
            menu_text = "\n".join(
                [f"{item['Item']} - {item['Price']}tk - {item['Category']} - Spicy: {item['Spicy']}, Cheese: {item['Contains_Cheese']}"
                 for item in dataset]
            )
        else:
            menu_text = dataset.to_string(index=False)

        enriched_prompt = f"""
You are a helpful and funny restaurant assistant. A user has asked a question about the menu.

Here is the menu (partial view):
{menu_text}

User asked:
\"{input.message}\"

Respond conversationally using only the menu data and facts provided. If the answer cannot be found, say so humorously.
"""

        conversation.messages[-1]["content"] = enriched_prompt

        response = query_groq_api(conversation)

        conversation.messages.append({
            "role": "assistant",
            "content": response
        })

        return {
            "response": response,
            "conversation_id": input.conversation_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
