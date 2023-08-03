import numpy as np
import pybedtools
import csv
###pre processess Const exons

# Load data from the file
data = np.genfromtxt("../datasets/5'100.csv", delimiter=',', dtype=object, skip_header=1, usecols=range(14), encoding='cp1252')

start_ind = 2  # Index of the start column
end_ind = 3  # Index of the end column
count_ind = 4  # Index of the count column

filtered_data = []
column_order = [0, 5, 7, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13]

for i in range(len(data) - 1):  # Iterate until the second-to-last row
    pSkip = 1
    nSkip = 1
    current_row = data[i]
    count = int(current_row[count_ind].decode('cp1252'))  # Access count column of current row
    start = int(current_row[start_ind].decode('cp1252'))  # Access start column of current row
    end = int(current_row[end_ind].decode('cp1252'))  # Access end column of current row

    if count >= 4 and (end - start) >= 25:
        if i + 1 < len(data):
            nSkip = 0
            next_row = data[i + 1]
            next_start = int(next_row[start_ind].decode('cp1252'))  # Access start column of next row
        if i - 1 >= 0:
            pSkip = 0
            prev_row = data[i - 1]
            prev_end = int(prev_row[end_ind].decode('cp1252'))  # Access end column of next row

        if pSkip == 1 and (next_start - end) >= 80 or \
           nSkip == 1 and (start - prev_end) >= 80 or \
           (start - prev_end) >= 80 and (next_start - end) >= 80:

            new_values = [item.decode('cp1252') for item in current_row]
            new_values[5] = str(start - 75)
            new_values[6] = str(end - start)
            new_values[7] = str(end + 75)
            new_values = [new_values[i] for i in column_order]  # Reordering the columns
            filtered_data.append(new_values)
            
# Specify the path to the output CSV file
output_file = "SEQ_5'.csv"
# Count the total number of rows in the BED file
total_rows = sum(1 for _ in filtered_data)
fasta = '../../util/hg38.fa'
i=1

# Open the output file in write mode
with open(output_file, 'w') as f:
    writer = csv.writer(f)
    # Write the header row
    writer.writerow(['PID', 'CLASS', 'CLASSNAME', 'SEQ'])
    
    # Iterate over the filtered data
    for row in filtered_data:
        start = int(row[1])
        stop = int(row[2])
        strand = row[3]
        protNm = row[13]
        classname = "5' Alternative Splice Event"
        i+=1

        # Create a BedTool object for the interval
        interval = pybedtools.BedTool(f"{row[0]}\t{start}\t{stop}\t{strand}", from_string=True)

        # Extract the sequence for the interval
        seq = interval.sequence(fi=fasta, s=True)

        # Read the sequence from the generated sequence file
        with open(seq.seqfn) as seq_file:
            seq_data = seq_file.read().strip().splitlines()[1]  # Skip the header line
            seq_data = seq_data.upper()
        
        # Write the data row to the output file
        writer.writerow([protNm+' | '+ row[0], '5', classname, seq_data])

        # Print the sequences
        print(f"Processing row {i} out of {total_rows}")
        
# Print the output file name
print(f"Sequences output file saved as: {output_file}")
