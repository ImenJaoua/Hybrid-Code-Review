import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Declare global variables
train_required = pd.DataFrame()
valid_required = pd.DataFrame()

def transform_dataset():
    global train_required, valid_required  # Use global to modify outer variables
    
    # Load your dataset
    df = pd.read_csv('filtered_data_token_length.csv')
    
    print(f"Total examples: {len(df)}")
        # Check for rows where 'review' is NaN
    nan_rows = df[df['review'].isna()]

    # Output the result
    if nan_rows.empty:
        print("No rows with NaN in the 'review' column.")
    else:
        print(f"Found {len(nan_rows)} rows with NaN in the 'review' column.")
        print(nan_rows)
    # Delete rows where 'review' is NaN
    df = df.dropna(subset=['review'])

# Print the total number of examples after deletion
    print(f"Total examples after deleting rows with NaN in 'review': {len(df)}")
    # Define filtering rules
    pmd_rules  = [
        'Avoid excessively',
        'Unused static',
        'The method',
        'Avoid variables',
        'A method should',
        'This class',
        'Class comments',
        'Unused import',
        'High amount',
        'The constant name',
        'The String literal',
        'Each class',
        'Possible God',
        'Linguistics Antipattern',
        'Avoid using',
        'Protected method',
        'Assigning an',
        'Avoid long',
        'The class',
        'The initializer',
        'The field name',
        'Avoid empty',
        'the final field',
        'Substitute calls', 
        'The value assigned',
        'The static field',
        'The public constant',
        'The JUnit ',
        'JUnit',
        'Enum',
        'This if',
        'Deeply nested',
        'When doing',
        'Avoid short',
        'The enum',
        'The constructor',
        'A getX()',
        'Too many',
        'The user-supplied',
        'The instance',
        "'catch' branch",
        'Using',
        'In JUnit4',
        'Consider',
        'Avoid protected',
        'This abstract class',
        'The native method',
        'No abstract',
        'The loop',
        'The annotation',
        'The utility',
        'This for',
        'The enum'
    ]

    # Define filtering rules for Checkstyle
    checkstyle_rules = [
        'Line',
        'Variable',
        'Missing',
        'Name',
        'First',
        'Distance',
        'Summary',
        'All overloaded',
        'More',
        'Using',
        'Parameter',
        'Local',
        'Javadoc',
        'Abbreviation',
        'Method',
        "'('" ,
        'Class',
        'Import',
        'Type',
        'Single-line',
        'Unused',
        'Block',
        'Must',
        'Interface',
        'Catch',
        'Unicode',
        'Forbidden',
        'Invalid'
    ]


    # Function to process rules
    def process_rules(rules, tool):
        global train_required, valid_required  # Declare global here
        for rule in rules:
            filtered_rows = df[(df['tool'] == tool) & (df['review'].str.startswith(rule))]
            
            if not filtered_rows.empty:
                if len(filtered_rows) < 10:
                    # If less than 10, add all to the training set
                    train_required = pd.concat([train_required, filtered_rows])
                else:
                    # Split the filtered rows into 80% train and 20% validation
                    train_subset, valid_subset = train_test_split(filtered_rows, test_size=0.2, random_state=42)
                    train_required = pd.concat([train_required, train_subset])
                    valid_required = pd.concat([valid_required, valid_subset])

    # Process PMD and Checkstyle rules
    process_rules(pmd_rules, 'PMD')
    process_rules(checkstyle_rules, 'Checkstyle')

    # Drop the required rows from the original dataset
    remaining_df = df.drop(train_required.index).drop(valid_required.index)

    # Split the remaining data into training and validation sets
    train_remaining, valid_remaining = train_test_split(remaining_df, test_size=0.2, random_state=42, stratify=remaining_df['tool'])

    # Combine the required rows with the remaining training and validation data
    train = pd.concat([train_remaining, train_required])
    valid = pd.concat([valid_remaining, valid_required])

    # Shuffle the training and validation data (optional)
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    temp_df = valid.sample(frac=1, random_state=42).reset_index(drop=True)

    # Step 1: Perform stratified splitting using pandas and sklearn
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['tool'])

    # Step 2: Convert DataFrames back to Dataset objects
    train_dataset = Dataset.from_pandas(train)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Combine into a single DatasetDict
    dataset_splits = DatasetDict({
        "train": train_dataset,
        "validation": valid_dataset,
        "test": test_dataset
    })

    # Print counts for each tool in each split
    print("Counts of each tool in the datasets:")
    for split in dataset_splits:
        df_split = dataset_splits[split].to_pandas()
        counts = df_split['tool'].value_counts()
        print(f"\n{split.capitalize()} Split:")
        print(counts)

    # Processing function
    def process_example(example):
        text_user = (
            f"Please analyze the following code change and provide a review, listing all potential issues, bugs, or areas for improvement. "
            f"Be specific and comprehensive in your feedback.\n\n"
            f"### Code Patch:\n{example['patch']}\n\n"
            f"### Review:"
                                    )
        text_assistant = example["review"]

        messages = [
            {"role": "user", "content": text_user},
            {"role": "assistant", "content": text_assistant}
        ]
        

        return {"messages": messages}

    # Apply transformation to all splits
    dataset_splits = dataset_splits.map(process_example, num_proc=8)

    # Save the dataset to disk
    dataset_splits.save_to_disk("processed_dataset_new_prompt")

    print("Dataset saved to disk.")

if __name__ == "__main__":
    transform_dataset()
