from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class ReasoningState (TypedDict):
    rigorousness_required :Annotated[bool, lambda x,y: y]

    messages :Annotated[list, add_messages]

    facts :Annotated[dict, lambda facts_a, facts_b: {**facts_a, **facts_b}]