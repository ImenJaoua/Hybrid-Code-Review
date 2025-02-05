llama_template = (
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "{system_prompt}<|eot_id|>\n"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{user_prompt}<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{{\"Review1\": "
)

review_classification_system_prompt = "You are an expert in coding and code review evaluation."

review_classification_template = """
[Task Overview]
You will review a Java code change along with five review comments. Your objective is to classify each review’s accuracy based on these categories:
    - Accurate
    - Partially Accurate
    - Not Accurate

[Evaluation Guidelines]
**Tie-Breaking**: If uncertain between "Accurate" and "Partially Accurate," default to "Accurate".
{patch}

[Reviews]
- **Review1**: {FTr}
- **Review2**: {RAG}
- **Review3**: {SA}
- **Review4**: {MLr}
- **Review5**: {concat}

[Classification Task]
Evaluate each review’s accuracy using the criteria above. Then provide your assessments in the following JSON format:

```json
{{
    "Review1": "<Correctness>",
    "Review2": "<Correctness>",
    "Review3": "<Correctness>",
    "Review4": "<Correctness>",
    "Review5": "<Correctness>"
}}

[Response] 
Provide your assessment using the JSON format specified above without any explanation.
"""
