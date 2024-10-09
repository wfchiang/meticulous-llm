from typing import Optional
from langchain_core.output_parsers import BaseOutputParser

import logging 
logging.basicConfig(level=logging.INFO)


# ====
# Parser helper functions 
# ====
def get_first_sentence (text :str) -> str: 
    assert(type(text) is str) 

    i_period = text.find(".")
    if (i_period < 0): 
        i_period = len(text)

    i_newline = text.find("\n")
    if (i_newline < 0): 
        i_newline = len(text)
    
    return text[0:min(i_period, i_newline)]


# ====
# Output parser classes 
# ====
class BooleanOutputParser (BaseOutputParser[bool]): 
    fallback_value :Optional[bool] = None 

    def parse (self, text :str) -> bool:
        try: 
            original_text = text 

            text = text.strip().lower()
            text = get_first_sentence(text) 

            if (text.find("yes") >= 0 or text.find("true") >= 0): 
                return True
            
            elif (text.find("no") >= 0 or text.find("false") >= 0): 
                return False 
            
            else: 
                raise Exception(f"Unable to parse output: {original_text}")

        except Exception as err:
            logging.error(err)
            if (type(self.fallback_value) is bool): 
                return self.fallback_value
            else: 
                raise err 


