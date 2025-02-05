import pandas as pd
import numpy as np
import subprocess
import re
import os 
# Assuming your CSV file is named 'output.csv' and is located in the current directory
file_path = 'results/results1.csv'
output_file = 'results71111.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
# print(df)
# # Specify the starting and ending index values
# start_index = 19983
# end_index = 20000

# # Find the position of the first occurrence of the start index
# start_position = df[df['Index'] == start_index].index[0]

# # Find the position of the last occurrence of the end index
# end_position = df[df['Index'] == end_index].index[-1]

# # Extract rows from the start to end positions
# df = df.loc[start_position:end_position]
print(df)
# Group by the 'Index' column and aggregate the 'review' column into lists
# Assuming df is your DataFrame with 'Index', 'review', and 'Code' columns
merged_reviews = df.groupby('Index').agg({
    'review': lambda x: list(x),   # Combine reviews into a list
    'code': 'first'     # Combine codes into a list (in case they differ)
}).reset_index()
print(merged_reviews)

# Function to process each Index
def rate_reviews(reviews_list,code):

    # Construct the prompt with the given code and reviews
    prompt = f"""
    [Task Description]
    You are given a piece of Java code and several code reviews. 
    Your task is to evaluate each review based on its relevance and accuracy to the provided code. 
    You will rate each review on a scale of 1 to 10. 

    [code]
        {code}

    [reviews]
    """
    for i, review in enumerate(reviews_list):
        prompt += f"""
        review {i} : {review}
        """
    prompt += """
    [Task]
    Provide your rating for each review. Your response format should be like this:
    Rating for review n: [rating]
    write this after continue rating: [End]

    [Response]
    """   
    if len(reviews_list) < 50:

        # Define the command
        command = [
            "python", "test_inference.py",
            "-m", "Llama-3-8B-Instruct-exl2"
        ]
         # Set environment variables to use a specific GPU (e.g., GPU 0)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "3"  # Change "0" to the GPU you want to use

        # Run the command with prompt passed through stdin and specific GPU
        result = subprocess.run(command, input=prompt, capture_output=True, text=True, env=env)
        print("Subprocess output:", result.stdout)
        
        # Extract ratings
        match = re.search(r"Extracted Ratings: (\[.*?\])", result.stdout)
        if match:
            list_string = match.group(1)
            extracted_list = eval(list_string)
            print("Extracted Ratings:", extracted_list)

            if len(extracted_list) == len(reviews_list):
                return extracted_list
            else:
                print(f"Length mismatch: {len(extracted_list)} ratings for {len(reviews_list)} reviews")
                return [np.nan] * len(reviews_list)
        else:
            print("Pattern not found in subprocess output")
            return [np.nan] * len(reviews_list)
    else:
         return [np.nan] * len(reviews_list)


# Write ratings to the output file immediately
for idx in merged_reviews.index:
    index_value = merged_reviews.loc[idx, 'Index']
    print(index_value)
    reviews_list = merged_reviews.loc[idx, 'review']
    code= merged_reviews.loc[idx, 'code']
    ratings = rate_reviews(reviews_list,code)
    
    # Extract the rows corresponding to this Index
    index_rows = df[df['Index'] == index_value]
    
    # If ratings are not NaN, update the 'rate' column and append to the output file
    if not pd.isna(ratings).all():
        for i, rating in enumerate(ratings):
            row_copy = index_rows.iloc[i].copy()
            row_copy['rate'] = rating
            
            # Check if output file already exists
            if os.path.exists(output_file):
                # Append without headers
                row_copy.to_frame().T.to_csv(output_file, mode='a', header=False, index=False)
            else:
                # Write with headers
                row_copy.to_frame().T.to_csv(output_file, index=False)

print("Processing completed")