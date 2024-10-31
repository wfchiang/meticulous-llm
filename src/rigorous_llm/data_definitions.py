from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class ReasoningState (TypedDict):
    rigorousness_required :Annotated[bool, lambda x,y: y]

    messages :Annotated[list, add_messages]

    facts_collected :Annotated[bool, lambda x,y: y]
    facts :Annotated[dict, lambda facts_a, facts_b: {**facts_a, **facts_b}]

    statements_extracted :Annotated[bool, lambda x,y: y]
    extracted_statements :Annotated[list, lambda es_a, es_b: es_b] 

    # These validated statements are those extracted from the last AIMessage and were passed the validation against the facts. 
    validated_statements :Annotated[list, lambda vs_a, vs_b: vs_b] # We overwrite the previous set of statements 


def collect_facts_from_state (state :ReasoningState) -> List[str]: 
    all_facts = [] 
    for facts in state["facts"].values(): 
        all_facts += facts
    return list(set(all_facts))