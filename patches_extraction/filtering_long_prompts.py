import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, EarlyStoppingCallback

# Load the dataset using pandas
df = pd.read_csv('filtered_data_token_length.csv')
print(len(df))
# model_name = "codellama/CodeLlama-7b-Instruct-hf"

#     # Initialize the tokenizer (replace 'your-model-name' with the appropriate model name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

#     # Tokenization and length calculation function
# def get_token_length(patch, review):
#         text_user = f"Generate a review comment for this code change.\n\n### Patch: \n\n{patch}\n\n### Review:"
#         text_assistant = review
#         full_text = f"{text_user}\n\n{text_assistant}"
#         tokens = tokenizer.encode(full_text, truncation=False)
#         return len(tokens)

#     # Apply tokenization and filter based on token length
# df['prompt_length'] = df.apply(lambda row: get_token_length(row['patch'], row['review']), axis=1)
# df = df[df['prompt_length'] < 2048]
# print(df)
# df.to_csv('results/filtered_data_token_length.csv', index=False)

#import pandas as pd

# # Function to filter based on Index and tool conditions
# def filter_indexes(file_path):
#     # Load the CSV file
#     df = pd.read_csv(file_path)
    
#     # Filter Indexes that are repeated at least twice
#     index_counts = df['Index'].value_counts()
#     repeated_indexes = index_counts[index_counts >= 2].index

#     # Filter rows where tool contains both 'codellama' and != 'codellama' for each repeated Index
#     filtered_df = df[df['Index'].isin(repeated_indexes)]
    
#     # Group by Index and filter those that have both 'codellama' and != 'codellama' tools
#     valid_indexes = filtered_df.groupby('Index').filter(
#         lambda group: 'CodeLlama' in group['tool'].values and any(group['tool'] != 'CodeLlama')
#     )

#     return valid_indexes

# # Example usage (replace 'path_to_csv_file.csv' with your actual file path)
# file_path = 'results/filtered_data_token_length.csv'  # Replace with actual file path
# filtered_indexes = filter_indexes(file_path)
# print(filtered_indexes.iloc[0])
# print(filtered_indexes.iloc[1])
