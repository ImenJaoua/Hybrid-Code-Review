import os
import subprocess

checkstyle_path = ["java", "-jar", "C:/Users/Lenovo/Downloads/checkstyle-10.17.0-all.jar"]
checkstyle_config = "google_checks2.xml"
java_folder = "C:/Users/Lenovo/Desktop/projet/final_formatted_java_code"
output_folder = "final_checkstyle_output3"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a sorted list of Java files in the folder
java_files = [file_name for file_name in os.listdir(java_folder) if file_name.endswith(".java")]

# Iterate through each Java file
for file_name in java_files:
    # Generate output file name based on input Java file with XML extension
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(file_name))[0] + ".txt")
    
    # Check if the output file already exists
    if not os.path.exists(output_file):
        # Construct the command with XML format
        command = ["java", "-jar", "C:/Users/Lenovo/Downloads/checkstyle-10.17.0-all.jar", "-c", checkstyle_config, "-o", output_file, os.path.join(java_folder, file_name)]
        
        # Run the command and redirect the output to the specific output file
        with open(output_file, "w") as f:
            subprocess.run(command, stdout=f)

print("Checkstyle analysis completed.")
