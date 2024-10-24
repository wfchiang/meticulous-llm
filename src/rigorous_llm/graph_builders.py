import os 
import json 
from typing import Dict
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langchain_core.runnables.base import Runnable
from langchain.chat_models.base import BaseChatModel 
from langgraph.graph import StateGraph, START, END

from .llms import create_default_openai_llm
from .data_definitions import ReasoningState, collect_facts_from_state
from .chains import create_chain_for_rigorousness_judgement, create_chain_for_statements_extraction, create_chain_for_input_validation_against_facts, create_chain_for_statements_summarization
from .utils import encode_text_list_to_bulleted_paragraph

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
            "facts": new_facts
        }
    
class LLMResponseValidationNode: 

    name :str = "llm_response_validation" 

    def __init__(
            self, 
            chat_model :BaseChatModel = create_default_openai_llm()
    ) -> None:
        self.chain_4_statements_extraction = create_chain_for_statements_extraction(llm=chat_model)
        self.chain_4_text_validation_against_facts = create_chain_for_input_validation_against_facts(llm=chat_model)

    def __call__(self, state :ReasoningState) -> ReasoningState:
        all_facts = collect_facts_from_state(state)
    
        if (len(all_facts) == 0): 
            # If there is no fact, then, there is no validated statement 
            logging.info(f"No fact, no validated statement.")

            # return 
            return {
                "validated_statements": [] 
            } 

        else: 
            # Find out the last AI message 
            last_ai_message = None 
            for i in range(len(state["messages"])-1, -1, -1): 
                if (isinstance(state["messages"][i], AIMessage) and state["messages"][i].content.strip() != ""): 
                    last_ai_message = state["messages"][i]
                    break 

            assert(last_ai_message is not None), "AIMessage not found"
            
            # Extract statements from the last AIMessage 
            extracted_statements = self.chain_4_statements_extraction.invoke({"input": last_ai_message.content})

            logging.info(f"{len(extracted_statements)} statements extracted from the last AI message")
            
            # validate the extracted statements 
            encoded_facts = encode_text_list_to_bulleted_paragraph(
                text_list=all_facts
            )

            validated_statements = [] 

            for es in extracted_statements: 
                judgement = self.chain_4_text_validation_against_facts.invoke({
                    "input": es, 
                    "facts": encoded_facts
                })

                if (judgement): 
                    validated_statements.append(es)

            logging.info(f"{len(validated_statements)} statements passed the validation")

            # return 
            return {
                "validated_statements": validated_statements
            }

class LLMResponseRevisementNode: 

    name :str = "llm_response_revisement"

    def __init__(
            self, 
            chat_model :BaseChatModel = create_default_openai_llm()
    ) -> None:
        self.chain_4_statements_summarization = create_chain_for_statements_summarization(llm=chat_model)

    def __call__(self, state :ReasoningState) -> ReasoningState:
        new_messages = [
            HumanMessage("Please revise rigorously") 
        ]

        validated_statements = state["validated_statements"]
        if (len(validated_statements) == 0): 
            new_messages.append(
                AIMessage("Sorry, I cannot answer it rigorously...")
            )

        else: 
            new_messages.append(
                AIMessage(self.chain_4_statements_summarization.invoke({
                    "statements": encode_text_list_to_bulleted_paragraph(text_list=validated_statements)
                }))
            )

        return {
            "messages": new_messages, 
            "rigorousness_required": False, # reset rigorousness_required flag
            "validated_statements": [] 
        } 
        
def rigorousness_judgement_conditional_edge (
        state :ReasoningState
):
    if (state["rigorousness_required"]): 
        return FactsCollectionNode.name 
    else: 
        return END


# ====
# Rigorous LLM graph 
# ====
def create_rigorous_llm_graph (chatbot_subgraph :StateGraph) -> StateGraph: 
    graph_builder = StateGraph(ReasoningState) 

    graph_builder.add_node(KEY_CHATBOT_SUBGRAPH, chatbot_subgraph.compile())
    graph_builder.add_node(RigorousnessJudgementNode.name, RigorousnessJudgementNode())
    graph_builder.add_node(FactsCollectionNode.name, FactsCollectionNode())
    graph_builder.add_node(LLMResponseValidationNode.name, LLMResponseValidationNode())
    graph_builder.add_node(LLMResponseRevisementNode.name, LLMResponseRevisementNode())

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

    graph_builder.add_edge(FactsCollectionNode.name, LLMResponseValidationNode.name)
    graph_builder.add_edge(LLMResponseValidationNode.name, LLMResponseRevisementNode.name)
    graph_builder.add_edge(LLMResponseRevisementNode.name, END)

    # return 
    return graph_builder