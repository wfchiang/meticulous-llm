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
from chains import create_chain_for_judging_the_need_of_reasoning 

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
            "messages": [chat_model_said]
        }

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    name :str = "default_casual_tools"

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: Dict):
        if messages := inputs.get("messages", []):
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
        return {"messages": outputs}
    
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
class JudgeTheNeedOfRigorousnessNode: 
    
    name :str = "judge_the_need_of_rigorousness"
    message_rigorousness_required = "Rigorousness required"
    message_no_rigorousness_required = "Just casual chat"

    def __init__ (
            self, 
            chat_model :Runnable = create_default_openai_llm()
    ): 
        self.chat_model = chat_model
        self.chain_4_judging_the_need_of_reasoning = create_chain_for_judging_the_need_of_reasoning(llm=chat_model)

    def __call__ (self, state :ReasoningState) -> ReasoningState: 
        old_messages = state["messages"]
        assert(len(old_messages) > 0)

        last_user_message = old_messages[-1]
        judgement = self.chain_4_judging_the_need_of_reasoning.invoke({"input": last_user_message})
        assert(type(judgement) is bool)

        logging.info(f"Judgement of the need of rigorousness: {judgement}")

        return {
            "messages": [
                (
                    AIMessage(self.message_rigorousness_required)
                    if (judgement)
                    else AIMessage(self.message_no_rigorousness_required)
                )
            ]
        }
    
class ExtractFactsNode: 

    name :str = "extract_facts"

    def __init__(
            self, 
            chat_model :Runnable = create_default_openai_llm()
    ):
        pass

    def __call__(self, state :ReasoningState) -> ReasoningState:
        return state
        
def judge_the_need_of_rigorousness_conditional_edges (
        state :ReasoningState
):
    assert("messages" in state)

    last_ai_message = state["messages"][-1]
    assert(isinstance(last_ai_message, AIMessage)), f"Expected an AI message but got {last_ai_message}"

    if (last_ai_message.content == JudgeTheNeedOfRigorousnessNode.message_rigorousness_required): 
        return END 
    else: 
        return ExtractFactsNode.name


# ====
# Rigorous LLM graph 
# ====
def create_rigorous_llm_graph (chatbot_subgraph :StateGraph) -> StateGraph: 
    graph_builder = StateGraph(ReasoningState) 

    graph_builder.add_node(KEY_CHATBOT_SUBGRAPH, chatbot_subgraph.compile())
    graph_builder.add_node(JudgeTheNeedOfRigorousnessNode.name, JudgeTheNeedOfRigorousnessNode())
    graph_builder.add_node(ExtractFactsNode.name, ExtractFactsNode())

    graph_builder.add_edge(START, KEY_CHATBOT_SUBGRAPH)
    graph_builder.add_edge(KEY_CHATBOT_SUBGRAPH, JudgeTheNeedOfRigorousnessNode.name)

    graph_builder.add_conditional_edges(
        JudgeTheNeedOfRigorousnessNode.name, 
        judge_the_need_of_rigorousness_conditional_edges, 
        {
            ExtractFactsNode.name: ExtractFactsNode.name, 
            END: END
        }
    )

    graph_builder.add_edge(ExtractFactsNode.name, END)

    # return 
    return graph_builder