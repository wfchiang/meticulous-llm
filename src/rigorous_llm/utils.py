from typing import List, Union 
from langchain_core.messages import BaseMessage


def encode_text_list_to_bulleted_paragraph (text_list :List[str], bullet_str :str = "*") -> str: 
    return "\n".join(
        list(map(
            lambda x: f"{bullet_str} {x.lstrip(bullet_str)}", 
            text_list
        ))
    )


def find_last_chat_message (
        message_list :List[BaseMessage], 
        message_type :BaseMessage, 
        return_none_for_not_found :bool = False 
) -> Union[BaseMessage, None]: 
    last_message_of_type = None 
    for i in range(len(message_list)-1, -1, -1): 
        pointed_message = message_list[i]
        if (isinstance(pointed_message, message_type) and pointed_message.content.strip() != ""): 
            last_message_of_type = pointed_message
            break 

    if (last_message_of_type is None and (not return_none_for_not_found)): 
        assert(last_message_of_type is not None), f"The targeted message type ({message_type}) not found"

    return last_message_of_type