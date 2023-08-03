import numpy as np
import pandas as pd

# Define the desired number of samples per label
num_samples_per_label = 2500

# Read the datasets
datasets = ['../datasets/Split2.0/SEQ_CONST.csv', 
            '../datasets/Split2.0/SEQ_ES.csv', 
            "../datasets/Split2.0/SEQ_3'.csv", 
            "../datasets/Split2.0/SEQ_5'.csv"]

# Combine all datasets into one DataFrame
combined_data = pd.concat([pd.read_csv(f) for f in datasets])
print(combined_data.shape)

# Count the occurrences of each label
label_counts = combined_data['CLASS'].value_counts()

# Initialize an empty DataFrame for the sampled data
sampled_data = pd.DataFrame()

# Iterate over each label
for label in label_counts.index:
    # Filter the combined data for the current label
    label_data = combined_data[combined_data['CLASS'] == label]
    
    # Check if the label has enough samples
    if len(label_data) >= num_samples_per_label:
        # Sample the desired number of samples per label
        sampled_label_data = label_data.sample(num_samples_per_label, random_state=42)
    else:
        # If the label has fewer samples, duplicate the existing samples to reach the desired number
        num_duplicates = num_samples_per_label // len(label_data)
        remainder = num_samples_per_label % len(label_data)
        duplicated_data = pd.concat([label_data] * num_duplicates + [label_data.sample(remainder, random_state=42)])
        sampled_label_data = duplicated_data.sample(frac=1, random_state=42)  # Shuffle the duplicated data
    
    # Concatenate the sampled label data to the overall sampled data
    sampled_data = pd.concat([sampled_data, sampled_label_data])

# Create a new DataFrame for the subset with the same class distribution
subset_data = sampled_data.copy()

# Save the subset data to a CSV file
subset_data.to_csv('../tNt/10ksubset_data.csv', index=False)
