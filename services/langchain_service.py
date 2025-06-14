import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        k=5  
    )

def create_vector_store():
    with open('./Databases/menues.json', 'r', encoding='utf-8') as f:
        data = json.load(f)  

    documents = [
        Document(
            page_content=f"Item: {item['Item']}, Category: {item['Category']}, Price: {item['Price']}, Spicy: {item['Spicy']}, Contains_Cheese: {item['Contains_Cheese']}, Vegetarian: {item['Vegetarian']}, Gluten_Free: {item['Gluten_Free']}, Availability: {item['Availability']}, Prep_Time_Min: {item['Prep_Time_Min']}, Calories: {item['Calories']}, Popular: {item['Popular']}",
            metadata={"source": "menu.json"}
        )
        for item in data
    ]

    document = "\n".join(doc.page_content for doc in documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     
        chunk_overlap=100,   
    )

    chunks = splitter.create_documents([document])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("Databases/faiss_index")

def retrieve_context(query):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("Databases/faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    retrieved_doc = retriever.invoke(query)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_doc)

    return context_text

def generate_response(query, context_text):
    prompt = PromptTemplate.from_template("""
    You are a friendly restaurant assistant.
    Respond conversationally using only the menu data and facts provided. 
    Response with concise and humble messages. 
    Convince user to order something from your restaurent. 
    Currency will be bangladeshi taka.

    Context:
    {context}

    Chat History:
    {history}

    User: {input}
    """)

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.6,
        groq_api_key=GROQ_API_KEY
    )

    def add_context_and_history(input_dict):
        input_text = input_dict["input"]
        chat_history = memory.load_memory_variables({})["history"]
        return {
            "input": input_text,
            "context": context_text,
            "history": chat_history
        }

    chain = (
        RunnableLambda(add_context_and_history)
        | prompt
        | llm
        | StrOutputParser()
    )

    memory.save_context({"input": query}, {"output": "placeholder"})

    response = chain.invoke({"input": query})

    memory.chat_memory.messages[-1].content = response

    return response
