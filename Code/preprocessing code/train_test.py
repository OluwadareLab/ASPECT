from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import os
############################################################################
##Combines datasets into a single data file and can split data 
##into testing and training data so that it can be held out till final test
############################################################################
# results_dir = Path(f"./")/ "graphs"

def count_plot(x, title, dir):
    label_counts = Counter(x)
    label_counts_df = pd.DataFrame.from_dict(label_counts, orient='index').reset_index()
    label_counts_df.columns = ['Label', 'Count']
    sns.barplot(x='Label', y='Count', data=label_counts_df, palette='Set4')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title(f'{title}')
    pathf = dir / 'graphs'
    os.makedirs(pathf, exist_ok=True)
    filename = f'{title}.png'
    plt.savefig(os.path.join(pathf, filename) )
    plt.show()
    
    
    
# List of dataset file names
# datasets = ['../datasets/Split2.0/SEQ_CONST.csv', 
#             '../datasets/Split2.0/SEQ_ES.csv', 
#             "../datasets/Split2.0/SEQ_3'.csv", 
#             "../datasets/Split2.0/SEQ_5'.csv"]

datasets = [
            # './backwards/decoded_3data.csv',
            # './backwards/decoded_5data.csv'
            # "../datasets/Split2.0/SEQ_3'.csv", 
            # "../datasets/Split2.0/SEQ_5'.csv",
            # './backwards/trunct_3data.csv',
            # './backwards/trunct_5data.csv',
            "../datasets/estSplit/SEQ_3'.csv",
            # "../datasets/estSplit/SEQ_5'.csv",
            "../datasets/estSplit/SEQ_CON.csv"
            ]

# Combine all datasets into one DataFrame
combined_data = pd.concat([pd.read_csv(f) for f in datasets])

combined_data["CLASS"] = combined_data["CLASS"].astype(int)

# Split the combined dataset into training and testing sets
train_data, test_data = train_test_split(
    combined_data, test_size=0.2, shuffle=True, 
    stratify=combined_data['CLASS'])

print(combined_data.shape)

# count_plot(train_data["CLASS"].tolist(), "Training Class Distribution", results_dir)
# count_plot(test_data["CLASS"].tolist(), "Testing Class Distribution", results_dir)

# Save the training and testing sets as separate CSV files
train_data.to_csv('../tNt/train_est.csv', index=False)
test_data.to_csv('../tNt/test/test_est.csv', index=False)
print(train_data.shape)
print(test_data.shape)