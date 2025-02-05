import os
import subprocess

pmd_path = "C:/Users/Lenovo/Desktop/pmd-bin-7.0.0-rc4/bin/pmd.bat"
java_folder = "C:/Users/Lenovo/Desktop/projet/final_formatted_java_code"
ruleset_path = "C:/Users/Lenovo/Desktop/pmd-bin-7.0.0-rc4/ruleset.xml"
output_folder = "final_pmd_output"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a sorted list of Java files in the folder
java_files = [file_name for file_name in os.listdir(java_folder) if file_name.endswith(".java")]

# Iterate through each Java file
for file_name in java_files:
    # Generate output file name based on input Java file
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(file_name))[0] + ".xml")
    
    # Construct the command
    command = [pmd_path, "check", "-d", os.path.join(java_folder, file_name), "-f", "xml", "-R", ruleset_path]
    
    # Run the command and redirect the output to the specific output file
    with open(output_file, "w") as f:
        subprocess.run(command, stdout=f)

print("PMD analysis completed.")
