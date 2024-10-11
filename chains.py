from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables.base import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models.base import BaseChatModel 

from output_parsers import BooleanOutputParser, StrListOutputParser


def create_chain_for_rigorousness_judgement (llm :BaseChatModel) -> Runnable: 
    """
    Given a LLM, create a chain to judge if the given input (string/text) needs a rigorous LLM response
    """
    prompt_template = PromptTemplate.from_template(
        """You'll need to decide whether the user query requires a rigorous answer. It means the query must be answered with truth. You must simply answer "true" or "false". No explanation of your answer. Here are some examples: 

User Query: 
{input}

Answer: 
"""
    )

    return (prompt_template | llm | BooleanOutputParser()) 


def create_chain_for_statements_extraction (llm :BaseChatModel) -> Runnable: 
    """
    Given a LLM, create a chain to extract statements from the given input (string/text)
    """
    prompt_template = PromptTemplate.from_template(
        """You will break a text (given under 'Text:') into several short but self-contained statements. Each statement captures a piece of fact in the text. If an empty text is provided, answer with an empty. Here is an example: 

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


def create_chain_for_input_validation_against_facts (llm :BaseChatModel) -> Runnable: 
    prompt_template = PromptTemplate.from_template(
        """You will validate a given text (under 'Text:') against all given facts (under 'Fact:'). If the text satisfies one fact but fails another, you will answer "false". If no facts were provided, you must answer "false". You must reply simply "true" or "false". No further explanation for your answer. Here are some examples: 

Text: 
John is a software engineer who focuses on developing machine learning applications. 

Facts: 
* John is a software engineer. 
* John focuses on developing machine learning applications. 

Answer: 
true 

Text: 
Adam works very hard. He works seven days a week. 

Facts: 
* Adam works very hard. 
* Adam works five days a week. 

Answer: 
false 

Text:
David is smart. 

Facts: 

Answer:
false

Now, validate the following text with the facts provided as follows. 

Text: 
{input} 

Facts: 
{facts}

Answer:
"""
    )

    return (prompt_template | llm | BooleanOutputParser())


def create_chain_for_statements_summarization (llm :BaseChatModel) -> Runnable: 
    prompt_template = PromptTemplate.from_template(
        """Summarize the statements provided under 'Statements:'. Your writeup must only include the provided statements. Do not introduce external knowledge. 

Statements: 
{statements}

Summary:
"""
    )

    return (prompt_template | llm | StrOutputParser())