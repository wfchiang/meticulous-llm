import os 
from langchain.chat_models.base import BaseChatModel
from langchain_openai.chat_models import ChatOpenAI


def create_default_openai_llm () -> BaseChatModel: 
    assert("OPENAI_API_KEY" in os.environ)
    assert("OPENAI_MODEL" in os.environ)

    return ChatOpenAI(
        model=os.environ["OPENAI_MODEL"], 
        api_key=os.environ["OPENAI_API_KEY"], 
        temperature=0.01
    )