import subprocess
import pandas as pd
import os
import tempfile
import shutil
import subprocess
import pandas as pd
import os
import tempfile
import shutil
import numpy as np
# Load data from .npy file
data = np.load('dataset-v2.npy', allow_pickle=True)

# Convert NumPy array to DataFrame
df = pd.DataFrame(data)

print(df.head())

# Output directory for saving the formatted Java code files
output_directory = 'final_formatted_java_code'
os.makedirs(output_directory, exist_ok=True)

# Function to format Java code using Google Java Format and save to file
def format_and_save_java_code(java_code, output_directory, filename_prefix):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(java_code)
        temp_file_path = temp_file.name
    
    try:
        subprocess.run(['java', '-jar', 'google-java-format.jar', temp_file_path, '--replace'], check=True)
        formatted_file_path = os.path.join(output_directory, f"{filename_prefix}.java")
        shutil.move(temp_file_path, formatted_file_path)  # Use shutil.move to rename/move files
        return formatted_file_path
    except subprocess.CalledProcessError as e:
        print(f"Error formatting Java code: {e}")
        return None
    except FileNotFoundError:
        print("Error: google-java-format.jar not found.")
        return None

# Apply changes to the new Java code and format each code block
for index, row in df.iterrows():
    java_code = row[0]['newf']
    formatted_file_path = format_and_save_java_code(java_code, output_directory, f"formatted_code_{index}")
    if formatted_file_path is not None:
        print(f"Formatted code saved to: {formatted_file_path}")
    else:
        print(f"Failed to format code at index {index}.")

