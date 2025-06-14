from fastapi import HTTPException, APIRouter, FastAPI
from schemas.userinput import UserInput
from contextlib import asynccontextmanager
from services.langchain_service import create_vector_store, retrieve_context, generate_response
from services.conversations import get_or_create_conversation

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_vector_store()
    yield

route = APIRouter(lifespan=lifespan)

@route.post("/chat/")
async def chat(input: UserInput):
    try:
        context = retrieve_context(input.message)
        response = generate_response(input.message, context)

        return {
            "response": response
        }
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")