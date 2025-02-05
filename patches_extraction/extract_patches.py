import numpy as np
import pandas as pd
import re 


# print("DataFrame has been updated and saved to 'filtered_data.csv'")
df = pd.read_csv('filtered_data_token_length.csv')

# filtered_df = df[(df['tool'] == 'PMD') & df['review'].str.match(r'^The String literal\b', case=False, na=False)]
# print(filtered_df.iloc[0]['patch'])
def extract_method(file_path, start_line):
    with open(file_path, 'r',encoding='utf-8') as file:
        lines = file.readlines()

    # To store the method lines
    method_lines = []
    in_method = False
    braces_count = 0
    

    for i, line in enumerate(lines):
        if i + 1 >= start_line:
            if not in_method:
                # Check if we are at the beginning of a method
                if '{' in line:
                    in_method = True
                    braces_count = line.count('{') - line.count('}')
                    method_lines.append(line)
                continue
            # Inside the method
            method_lines.append(line)
            braces_count += line.count('{')
            braces_count -= line.count('}')

            if braces_count == 0:
                break

    return method_lines
import re


import pandas as pd
import re

def extract_word_after_variable(text):
    # Define a regular expression pattern to match the word after the word 'variable'
    # and remove any surrounding quotes or extra characters
    pattern = r"\bvariable\b\s+['\"]?(\w+)['\"]?"
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    if match:
        # Return the word following 'variable', without quotes or extra characters
        return match.group(1)
    else:
        # Return a message if no word is found after 'variable'
        return "No word found after the word 'variable'."


def find_second_usage(file_path, review, declaration_line):
    # Read the lines from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    variable_name = extract_word_after_variable(review)
    # Print the declaration line
    declaration_line_content = lines[declaration_line - 1].strip()
    print(f"Declaration of '{variable_name}' is on line {declaration_line}: {declaration_line_content}")

    usage_count = 0
    second_usage_line = None

    # Find the usage lines after the declaration line
    for i in range(declaration_line, len(lines)):
        line = lines[i]
        if re.search(rf"\b{variable_name}\b", line):
            usage_count += 1
            if usage_count == 2:  # Changed from 1 to 2 for second usage
                second_usage_line = i + 1  # Return the line number, not the content
                break

    if second_usage_line:
        print(f"Second usage of '{variable_name}' is on line {second_usage_line}")
        return second_usage_line  # return the second usage line number
    else:
        if usage_count == 1:
            print(f"Only one usage of '{variable_name}' found after declaration.")
        else:
            print(f"No usage of '{variable_name}' found after declaration.")
        return None


def generate_patch(file_path, review, beginline, endline):
    # Ensure beginline and endline are valid numbers and convert to integers
    beginline = int(float(beginline)) if pd.notna(beginline) else None
    endline = int(float(endline)) if pd.notna(endline) else None
    
    # Read the lines from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    

    # Calculate the range for context lines before and after the target lines
    context_before = max(0, beginline -2)  #2 lines before beginline, 0-based index
    context_after = min(len(lines), endline +2)  # 2 lines after endline, ensure not to exceed file length

    # Extract lines with context
    extracted_lines = lines[context_before:context_after]

    # Generate a patch-like output
    patch = []
    patch.append(f"@@ -{beginline},{endline - beginline + 1} +{beginline},{endline - beginline + 1} @@\n")
    
    # Add the extracted lines to the patch
    for i, line in enumerate(extracted_lines, start=context_before + 1):
        if beginline <= i <= endline:
            patch.append(f"+{line}")  # Highlight lines within the review range
        else:
            patch.append(f" {line}")  # Context lines without a marker

    # Combine the patch list into a single string
    patch_output = ''.join(patch)
    return patch_output

# Apply the function to each row where the tool is 'pmd' and review contains the specified text
for i, row in df.iterrows():
    if pd.notna(row['beginline']) and pd.notna(row['endline']):
        if (row['tool'] == 'checkstyle') and row['review'].startswith("Forbidden") :

            patch = generate_patch(
                file_path=f"final_formatted_java_code/formatted_code_{row['Index']}.java",
                review=row['review'],
                beginline=row['beginline'],
                endline=row['endline'],
            )
            print("Patch generated.")

            df.at[i, 'patch'] = patch

df.to_csv('filtered_data_token_length.csv', index=False)
