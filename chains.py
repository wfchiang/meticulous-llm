from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables.base import Runnable
from langchain.chat_models.base import BaseChatModel 

from output_parsers import BooleanOutputParser, StrListOutputParser


def create_chain_for_rigorousness_judgement (llm :BaseChatModel) -> Runnable: 
    """
    Given a LLM, create a chain to judge if the given input (string/text) needs a rigorous LLM response
    """
    prompt_template = PromptTemplate.from_template(
        """You'll need to decide whether the user query or request requires rigorous reasoning and must be answered with truth. You must simply answer "true" or "false". No explanation of your answer. 

User Query: 
{input}
"""
    )

    return (prompt_template | llm | BooleanOutputParser()) 


def create_chain_for_statements_extraction (llm :BaseChatModel) -> Runnable: 
    """
    Given a LLM, create a chain to extract statements from the given input (string/text)
    """
    prompt_template = PromptTemplate.from_template(
        """You will break a text (given under 'Text:') into several short but self-contained statements. Each statement captures a piece of fact in the text. For example: 

Text: 
Google LLC is an American multinational corporation and technology company focusing on online advertising, search engine technology, cloud computing, and artificial intelligence (AI). Google was founded on September 4, 1998, by American computer scientists Larry Page and Sergey Brin while they were PhD students at Stanford University in California. Together, they own about 14% of its publicly listed shares and control 56% of its stockholder voting power through super-voting stock. 

Facts: 
* Google LLC is an American multinational corporation and technology company. 
* Google focuses on online advertising, search engine technology, cloud computing, and artificial intelligence (AI).
* Google was founded on September 4, 1998, by American computer scientists Larry Page and Sergey Brin. 
* Larry Page and Sergey Brin found Google while they were PhD students at Stanford University in California. 
* Larry Page and Sergey Brin together own about 14\% of Google's publicly listed shares. 
* Larry Page and Sergey Brin together control 56\% of its stockholder voting power through super-voting stock. 

Now, break the following text into short but self-contained statements. 

Text: 
{input}

Facts: 
"""
    )
    
    return (prompt_template | llm | StrListOutputParser())