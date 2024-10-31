# Rigorous LLM 

This is my very preliminary project of developing an LLM agent that only includes the truth or facts in the responses when a user's query requires rigorousness. 

I will use LangGraph to implement my "version alpha", but the idea/concept is not limited by LLM agent frameworks (e.g., LangGraph). 

## Key Idea 

The current method is illustrated by the following figure. 

![](./docs/workflow.png)


* **chatbot_subgraph**: This is just a "normal" chatbot which takes a user query and generates an LLM responses. 

* **rigorousness_judgement**: 
    - This is to judge if the user query requires "rigorousness". For example, user request like "tell me a story" does not require rigorousness. On the other hand, "What is Google LLC?" does. 
    - If the stage does not consider the rigorousness is required by the query, the workflow will just get to the "END" -- just returns **chatbot_subgraph** response. 

* **sub_tasks_launcher**: It launches two sub-tasks running in parallel: **fact_collection** and **llm_response_statements_extraction**. 

* **facts_collection**: This will collect "facts" in different ways. The current implementation is just extracting facts from the tool usages. It will be revised in the later versions.

* **llm_response_statements_extraction**: It will decompose the former generated LLM response (the output of **chatbot_subgraph**) into "statements". Each of the statement holds a piece of information in the response. 

* **llm_response_validation**: It will join the outputs of stages, **facts_collection** and **llm_response_statements_extraction**, and validate the extracted statements agaist the facts. 

* **llm_response_revisement**: For the statements pass the validation of **llm_response_validation**, they will be re-composed into a revised message. 

## Demo 

Please see my notebook [README.ipynb](./README.ipynb) as a "demo". 