import pandas as pd
import random

def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip() for line in file.readlines()[1:]]  # Skip the first row (header row)
    return data

def write_data_to_file(file_path, data):
    with open(file_path, 'w') as file:
        file.write('PID,CLASS,CLASSNAME,SEQ\n')
        for row in data:
            file.write(row + '\n')

def randomize_data(data):
    random.shuffle(data)
    return data

def truncate_data_to_match_other_file(data_to_truncate, other_data):
    num_rows_to_truncate = min(len(data_to_truncate), len(other_data))
    truncated_data = data_to_truncate[:num_rows_to_truncate]
    return truncated_data

if __name__ == "__main__":
    input_file1 = "../datasets/Split2.0/SEQ_3'.csv"   # File to be truncated and randomized
    input_file2 = './backwards/decoded_3data.csv'  # File with desired number of rows
    output_file = './backwards/trunct_3data.csv'  # Truncated and randomized data will be written here
    
        # Step 1: Read data from both files
    data_to_truncate = read_data_from_file(input_file1)
    other_data = read_data_from_file(input_file2)

    # Step 2: Randomize the data in both files
    randomized_data_to_truncate = randomize_data(data_to_truncate)
    randomized_other_data = randomize_data(other_data)

    # Step 3: Determine the number of rows to truncate
    num_rows_to_truncate = min(len(randomized_data_to_truncate), len(randomized_other_data))

    # Step 4: Truncate the data to match the desired number of rows
    truncated_data_to_truncate = truncate_data_to_match_other_file(randomized_data_to_truncate, randomized_other_data)

    # Step 5: Write the truncated and randomized data to a new file
    write_data_to_file(output_file, truncated_data_to_truncate)

    out_data = read_data_from_file(output_file)
    
    print("Shape of data_to_truncate:", len(data_to_truncate))
    print("Shape of other_data:", len(other_data))
    print("Shape of other_data:", len(out_data))
    
    input_file1 = "../datasets/Split2.0/SEQ_5'.csv"   # File to be truncated and randomized
    input_file2 = './backwards/decoded_5data.csv'  # File with desired number of rows
    output_file = './backwards/trunct_5data.csv'  # Truncated and randomized data will be written here

    # Step 1: Read data from both files
    data_to_truncate = read_data_from_file(input_file1)
    other_data = read_data_from_file(input_file2)

    # Step 2: Randomize the data in both files
    randomized_data_to_truncate = randomize_data(data_to_truncate)
    randomized_other_data = randomize_data(other_data)

    # Step 3: Determine the number of rows to truncate
    num_rows_to_truncate = min(len(randomized_data_to_truncate), len(randomized_other_data))

    # Step 4: Truncate the data to match the desired number of rows
    truncated_data_to_truncate = truncate_data_to_match_other_file(randomized_data_to_truncate, randomized_other_data)

    # Step 5: Write the truncated and randomized data to a new file
    write_data_to_file(output_file, truncated_data_to_truncate)

    out_data = read_data_from_file(output_file)
    
    print("Shape of data_to_truncate:", len(data_to_truncate))
    print("Shape of other_data:", len(other_data))
    print("Shape of other_data:", len(out_data))
    
# "../datasets/Split2.0/SEQ_3'.csv", 
#             "../datasets/Split2.0/SEQ_5'.csv"