import pandas as pd

def check_sequence_length(input_file):
    # Read the data from the CSV file into a pandas DataFrame
    data = pd.read_csv(input_file)
    count = 0
    # Iterate through each row and check the length of the "SEQ" column
    for index, row in data.iterrows():
        seq_length = len(row["SEQ"])
        if seq_length < 280:
            print(f"Row index {index} has a sequence length of {seq_length} characters, which is shorter than 280.")
            count += 1
    print(f"There are {count} rows which are shorter than 280.")
if __name__ == "__main__":
    input_file = "../tNt/10ksubset_data.csv"  # Replace with the path to your input CSV file

    # Call the function to check the sequence length and print row indices if they are shorter than 280
    check_sequence_length(input_file)
