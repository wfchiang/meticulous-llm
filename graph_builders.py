import os 
import json 
from typing import Dict, List, Optional
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.runnables.base import Runnable
from langchain.chat_models.base import BaseChatModel 
from langchain_community.tools import BaseTool
from langgraph.graph import StateGraph, START, END

from llms import create_default_openai_llm
from data_definitions import ReasoningState 
from chains import create_chain_for_rigorousness_judgement, create_chain_for_statements_extraction

import logging 
logging.basicConfig(level=logging.INFO)


# ====
# Constants 
# ====
KEY_CHATBOT_SUBGRAPH = "chatbot_subgraph"


# ====
# Basic graph nodes and conditional edges 
# ====
class BasicChatModelNode: 
    """A node that runs the LLM for the last HumanMessage."""

    name :str = "default_casual_chatbot"

    def __init__(self, chat_model :Runnable) -> None:
        self.chat_model = chat_model 

    def __call__(self, state :Dict):
        assert("messages" in state) 
        
        chat_model_said = self.chat_model.invoke(state["messages"])
        return {
            **state, 
            "messages": [chat_model_said], 
        }

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    name :str = "default_casual_tools"

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: Dict):
        if messages := state.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {
            **state, 
            "messages": outputs
        }
    
def basic_chat_model_conditional_edges(
    state: ReasoningState
):
    assert("messages" in state) 
    assert(len(state["messages"]) > 0)

    last_ai_message = state["messages"][-1]

    if hasattr(last_ai_message, "tool_calls") and len(last_ai_message.tool_calls) > 0:
        return BasicToolNode.name
    
    return END
    

# ====
# Default casual chatbot graph 
# ====
def create_default_casual_chatbot_graph_builder () -> StateGraph: 
    assert("TAVILY_API_KEY" in os.environ)

    # create the default LLM 
    llm = create_default_openai_llm() 

    # create a tavily search tool and bind it with the LLM 
    from langchain_community.tools.tavily_search import TavilySearchResults
    llm_tools = [
        TavilySearchResults(max_results=3) 
    ]
    llm = llm.bind_tools(llm_tools)

    # create a graph builder 
    graph_builder = StateGraph(ReasoningState)
    
    graph_builder.add_node(BasicChatModelNode.name, BasicChatModelNode(chat_model=llm))
    graph_builder.add_edge(START, BasicChatModelNode.name)
    
    graph_builder.add_node(BasicToolNode.name, BasicToolNode(tools=llm_tools))
    graph_builder.add_edge(BasicToolNode.name, BasicChatModelNode.name)

    graph_builder.add_conditional_edges(
        BasicChatModelNode.name, 
        basic_chat_model_conditional_edges, 
        {
            BasicToolNode.name: BasicToolNode.name, 
            END: END 
        }
    )

    # return 
    return graph_builder


# ====
# Rigorous LLM nodes and edges 
# ====
class RigorousnessJudgementNode: 
    
    name :str = "rigorousness_judgement"

    def __init__ (
            self, 
            chat_model :BaseChatModel = create_default_openai_llm()
    ): 
        self.chain_4_judging_the_need_of_reasoning = create_chain_for_rigorousness_judgement(llm=chat_model)

    def __call__ (self, state :ReasoningState) -> ReasoningState: 
        old_messages = state["messages"]
        assert(len(old_messages) > 0)

        last_user_message = old_messages[-1]
        judgement = self.chain_4_judging_the_need_of_reasoning.invoke({"input": last_user_message})
        assert(type(judgement) is bool)

        logging.info(f"Judgement of the need of rigorousness: {judgement}")

        return {
            **state, 
            "rigorousness_required": judgement
        }
    
class FactsCollectionNode: 

    name :str = "facts_collection"

    def __init__(
            self, 
            chat_model :BaseChatModel = create_default_openai_llm()
    ):
        self.chain_4_statements_extraction = create_chain_for_statements_extraction(llm=chat_model)

    def __call__(self, state :ReasoningState) -> ReasoningState:
        # Extracting facts from tool messages 
        new_facts = {} 
        for message in state["messages"]: 
            if (isinstance(message, ToolMessage) and message.id not in state["facts"]): 
                logging.info(f"Extracting facts from tool messages: {message.id}")
                extracted_statements = self.chain_4_statements_extraction.invoke({"input": message.content})
                new_facts[message.id] = extracted_statements
        
        # Return 
        return {
            **state, 
            "facts": {**state["facts"], **new_facts}
        }
    
class ValidateLLMResponse: 

    name :str = "validate_llm_response" 

    def __init__(self) -> None:
        pass

    def __call__(self, state :ReasoningState) -> ReasoningState:
        return state

class ReviseLLMResponse: 

    name :str = "revise_llm_response"

    def __init__(self) -> None:
        pass

    def __call__(self, state :ReasoningState) -> ReasoningState:
        return state 
        
def rigorousness_judgement_conditional_edge (
        state :ReasoningState
):
    if (state["rigorousness_required"]): 
        logging.info("Rigorousness required!")
        return FactsCollectionNode.name 
    else: 
        logging.info("Just a casual chatting...")
        return END


# ====
# Rigorous LLM graph 
# ====
def create_rigorous_llm_graph (chatbot_subgraph :StateGraph) -> StateGraph: 
    graph_builder = StateGraph(ReasoningState) 

    graph_builder.add_node(KEY_CHATBOT_SUBGRAPH, chatbot_subgraph.compile())
    graph_builder.add_node(RigorousnessJudgementNode.name, RigorousnessJudgementNode())
    graph_builder.add_node(FactsCollectionNode.name, FactsCollectionNode())
    graph_builder.add_node(ValidateLLMResponse.name, ValidateLLMResponse())
    graph_builder.add_node(ReviseLLMResponse.name, ReviseLLMResponse())

    graph_builder.add_edge(START, KEY_CHATBOT_SUBGRAPH)
    graph_builder.add_edge(KEY_CHATBOT_SUBGRAPH, RigorousnessJudgementNode.name)

    graph_builder.add_conditional_edges(
        RigorousnessJudgementNode.name, 
        rigorousness_judgement_conditional_edge, 
        {
            FactsCollectionNode.name: FactsCollectionNode.name, 
            END: END
        }
    )

    graph_builder.add_edge(FactsCollectionNode.name, ValidateLLMResponse.name)
    graph_builder.add_edge(ValidateLLMResponse.name, ReviseLLMResponse.name)
    graph_builder.add_edge(ReviseLLMResponse.name, END)

    # return 
    return graph_builder