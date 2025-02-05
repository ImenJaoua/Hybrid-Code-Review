import os
import xml.etree.ElementTree as ET
import pandas as pd
import re
import numpy as np

def read_code_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return file.read()

def parse_audit_comments(text, index, code):

    comments = []
    lines = text.splitlines()
    
    # Regular expression to match the required format
    pattern = re.compile(
        r'\[WARN\] (.+?):(\d+)(?::(\d+))?: (.+)$'
    )
    
    seen_comments = set()
    
    for line in lines:
        match = pattern.match(line)
        if match:
            line_number = match.group(2).strip()
            comment = match.group(4).strip()
            comment = comment.split(' [')[0]

            # Check if the comment has been seen before
            if comment not in seen_comments:
                seen_comments.add(comment)
                
                comments.append({
                    'Index': index,
                    'code': code,
                    'patch': None,  # This will convert to NaN in the DataFrame
                    'review': comment,
                    'beginline': line_number,
                    'endline': line_number,
                    'tool':"checkstyle"
                })
    
    return comments


def parse_pmd_xml(file_path, index, code):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        namespaces = {'pmd': 'http://pmd.sourceforge.net/report/2.0.0'}
        
        results = []
        seen_comments = set()

        for violation in root.findall(".//pmd:violation", namespaces):
            comment = violation.text.strip()
            beginline = violation.get('beginline')
            endline = violation.get('endline')
            if comment not in seen_comments:
                seen_comments.add(comment)
                
                results.append({
                    'Index': index,
                    'code': code,
                    'patch': None,
                    'review': comment,
                    'beginline': beginline,
                    'endline': endline,
                    'tool': "PMD"
                })
        
        return results
    
    except ET.ParseError as e:
        print(f"Error parsing XML file {file_path}: {e}")
        return []  # Return an empty list if parsing fails

    except Exception as e:
        print(f"An unexpected error occurred with file {file_path}: {e}")
        return []  # Return an empty list if any other error occurs


def extract_reviews_and_patches_from_array(loaded_arr, index,code):
    reviews = []
    
    # Use index to extract the specific line from loaded_arr
    item = loaded_arr[index]
        # Extract review from 'generated_response'
    if 'generated_response' in item:
            generated_responses = item['generated_response']
            for response in generated_responses:
                review = response.split("Code Review Comment:", 1)[-1].strip()
                # Extract patch if available
                patch = item.get('patch', None)
                # Add extracted review and patch to the list
                reviews.append({
                    'Index': index,
                    'code': code, 
                    'patch': patch,
                    'review': review,
                    'beginline': None,
                    'endline': None,
                    'tool': "CodeLlama"
                })
    
    return reviews
def extract_index_from_filename(filename):
    # Extract the index from the filename after 'formmated_code_' and before '.java'
    if filename.endswith('.java'):
        base_name = filename[:-5]  # Remove '.java'
        parts = base_name.split('_')
        if len(parts) > 2 and parts[-1].isdigit():
            return int(parts[-1])
    return None


def parse_multiple_files(folder, directory):
    all_results = []
         # Load the array from the .npy file
    loaded_arr = np.load('codellama_codereview_preds.npy', allow_pickle=True)
    for filename1 in os.listdir(folder):
        index = extract_index_from_filename(filename1)
        code_file_path = os.path.join(folder, filename1)
        code = read_code_from_file(code_file_path)
        
        for filename2 in os.listdir(directory):
            if filename1.split(".")[0] == filename2.split(".")[0]:
                if filename2.endswith(".xml"):
                    file_path = os.path.join(directory, filename2)
                    file_results = parse_pmd_xml(file_path, index, code)
                    all_results.extend(file_results)
                elif filename2.endswith(".txt"):
                    file_path = os.path.join(directory, filename2)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        audit_text = file.read()
                        comments2 = parse_audit_comments(audit_text,index,code)
                        all_results.extend(comments2)

   
         # Extract reviews and patches from the loaded array
        reviews = extract_reviews_and_patches_from_array(loaded_arr,index,code)
        
        # Append the extracted reviews to the results list
        all_results.extend(reviews)

            
    return all_results

folder = "final_formatted_java_code"
directory = "knowledge_based_systems"

# Example usage
parsed_results = parse_multiple_files(folder, directory)

# Convert results to a DataFrame
df = pd.DataFrame(parsed_results, columns=['Index', 'code', 'patch', 'review', 'beginline', 'endline','tool'])


# Assuming `df` is your DataFrame
# Step 1: Convert the DataFrame to a NumPy array
np_array = df.to_numpy()

# Step 2: Save the NumPy array to a .npy file
npy_file_path = 'output2.npy'  # Replace with your desired .npy file path
np.save(npy_file_path, np_array)
