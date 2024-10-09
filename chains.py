from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables.base import Runnable
from langchain.chat_models.base import BaseChatModel 

from output_parsers import BooleanOutputParser


def create_chain_for_judging_the_need_of_reasoning (llm :BaseChatModel) -> Runnable: 
    prompt_template = PromptTemplate.from_template(
        """You'll need to decide whether the user query or request requires rigorous reasoning and must be answered with truth. You must simply answer "true" or "false". No explanation of your answer. 

User Query: 
{input}
"""
    )

    return (prompt_template | llm | BooleanOutputParser()) 