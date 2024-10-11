from typing import List 


def encode_text_list_to_bulleted_paragraph (text_list :List[str], bullet_str :str = "*") -> str: 
    return "\n".join(
        list(map(
            lambda x: f"{bullet_str} {x.lstrip(bullet_str)}", 
            text_list
        ))
    )