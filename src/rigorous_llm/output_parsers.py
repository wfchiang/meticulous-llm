import re 
from typing import Optional, List, Union
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

def strip_and_remove_empty_strings_from_list (text_list :List[str]) -> List[str]: 
    text_list = list(map(lambda t: t.strip(), text_list))
    text_list = list(filter(lambda t: t!="", text_list))
    return text_list

def split_text_by_separators (text :str, separators :Union[List[str], str]) -> List[str]: 
    assert(type(text) is str) 

    if (type(separators) is str): 
        separators = [separators]
    assert(isinstance(separators, List) and all([type(sep) is str for sep in separators]))

    text_lines = [text]
        
    for sep in separators: 
        tmp_text_lines = [] 

        for tline in text_lines: 
            tmp_text_lines += tline.split(sep)
        
        text_lines = tmp_text_lines

    return text_lines


# ====
# Output parser classes 
# ====
class BooleanOutputParser (BaseOutputParser[bool]): 

    fallback_value :Optional[bool]= None 

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
            
    @property  
    def _type(self) -> str:  
        return "boolean_output_parser"

class StrListOutputParser (BaseOutputParser[List[str]]): 

    bullet_patterns :List[str] = [
        r"^\s*(\d+|[\*])([\.:\s]{0,1})"
    ]

    separators :List[str] = ["\n"]

    def parse (self, text :str) -> List[str]: 
        # break text into lines 
        text_lines = split_text_by_separators(text=text, separators=self.separators)

        # clean up
        text_lines = strip_and_remove_empty_strings_from_list(text_lines)

        # group lines by bullet patterns 
        text_groups = [[]]

        for tline in text_lines: 
            matched_bullet_header_end_idx = -1 
            
            for b_pattern in self.bullet_patterns: 
                re_match = re.match(b_pattern, tline)
                if (re_match is not None): 
                    matched_bullet_header_end_idx = re_match.span()[1]
                    break 
            
            if (matched_bullet_header_end_idx > 0): 
                text_groups.append([tline[matched_bullet_header_end_idx:].strip()])
            else:
                text_groups[-1].append(tline)

        text_groups = list(filter(lambda g: len(g) > 0, text_groups))

        # create the aggregated text list 
        aggregated_text_list = list(map(
            lambda tg: " ".join(tg), 
            text_groups
        )) 

        # clean up 
        aggregated_text_list = strip_and_remove_empty_strings_from_list(aggregated_text_list)

        # return 
        return aggregated_text_list

    @property  
    def _type(self) -> str:  
        return "str_list_output_parser"
    

class SelectionIndicesOutputParser(BaseOutputParser[int]): 

    bullet_patterns :List[str] = [
        r"^\s*(\d+)([\.:\s]{0,1})", 
        r"^\s*\[(\d+)\]([\.:\s]{0,1})", 
        r"^\s*([(\d+)\)([\.:\s]{0,1})"
    ]

    separators :List[str] = ["\n"]

    def parse (self, text :str) -> List[str]: 
        # break text into lines 
        text_lines = split_text_by_separators(text=text, separators=self.separators)

        # clean up
        text_lines = strip_and_remove_empty_strings_from_list(text_lines)

        # group lines by bullet patterns 
        selected_indices = []

        for tline in text_lines:     
            for b_pattern in self.bullet_patterns: 
                re_match = re.match(b_pattern, tline)
                if (re_match is not None): 
                    selected_indices.append(int(re_match.groups()[0])) 
                    break 

        # return 
        return selected_indices

    @property  
    def _type(self) -> str:  
        return "selection_indices_output_parser"
