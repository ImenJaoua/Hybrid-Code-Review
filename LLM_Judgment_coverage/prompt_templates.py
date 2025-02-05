llama_template = (
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "{system_prompt}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{user_prompt}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    " {{ Review1: "
)

review_classification_system_prompt = """You are an expert in coding and code peer-reviewing."""

review_classification_template = """[Task Description] 
You are given a Java code changes along with six responses:  
Your task is to rank these reviews from 1 to 5, where 1 is the most complete, and 5 is the least complete, with respect to the code change.

[Code change] 
{patch} 


[Review1] 
{FTr} 

[Review2] 
{MLr} 

[Review3] 
{concat} 

[Review4] 
{RAG} 

[Review5] 
{SA}

[Task] 
Provide your rankings in the following JSON format:

{{
    Review1: <rank>,
    Review2: <rank>,
    Review3: <rank>,
    Review4: <rank>,
    Review5: <rank>,
}}
Ensure that two reviews cant receive the same rank.
When ranking, focus on comparing the completness of the review.

[Response]
"""