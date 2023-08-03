import numpy as np
import pybedtools
import csv


# Load data from the file
data = np.genfromtxt("../datasets/strictEST/5'exon.csv", delimiter=',', dtype=object, skip_header=1, usecols=range(13), encoding='cp1252')

start_ind = 2  # Index of the start column
end_ind = 3  # Index of the end column
count_ind = 4  # Index of the count column

filtered_data = []
column_order = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12]

for i in range(len(data)):  # Iterate until the second-to-last row
    pSkip = 1
    nSkip = 1
    current_row = data[i]
    count = int(current_row[count_ind].decode('cp1252'))  # Access count column of current row
    start = int(current_row[start_ind].decode('cp1252'))  # Access start column of current row
    end = int(current_row[end_ind].decode('cp1252'))  # Access end column of current row

    if count >= 0 and (end - start) >= 25:
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
            new_values[6] = str(start + 75)
            new_values[7] = str(end - 75)
            new_values[8] = str(end + 75)
            new_values = [new_values[i] for i in column_order]  # Reordering the columns
            filtered_data.append(new_values)
            
# Specify the path to the output CSV file
output_file = "SEQ_5'.csv"
# Count the total number of rows in the BED file
total_rows = sum(1 for _ in filtered_data)
fasta = '../../util/hg38.fa'
i=0

# Open the output file in write mode
with open(output_file, 'w') as f:
    writer = csv.writer(f)
    # Write the header row
    writer.writerow(['CLASS', 'SEQ'])
    
    # Iterate over the filtered data
    for row in filtered_data:
        start1 = int(row[1])
        stop1 = int(row[2])
        start2 = int(row[3])
        stop2 = int(row[4])
        strand = row[5]
        protNm = row[12]
        classname = "5' Alternative Splice Event"
        i+=1

        # Create a BedTool object for the interval
        interval1 = pybedtools.BedTool(f"{row[0]}\t{start1}\t{stop1}\t{strand}", from_string=True)
        interval2 = pybedtools.BedTool(f"{row[0]}\t{start2}\t{stop2}\t{strand}", from_string=True)
        # Extract the sequence for the interval
        seq1 = interval1.sequence(fi=fasta, s=True)
        seq2 = interval2.sequence(fi=fasta, s=True)

        # Read the sequence from the generated sequence file
        with open(seq1.seqfn) as seq1_file, open(seq2.seqfn) as seq2_file:
            seq1_data = seq1_file.read().strip().splitlines()[1]  # Skip the header line
            seq2_data = seq2_file.read().strip().splitlines()[1]  # Skip the header line
            seq1_data = seq1_data.upper()
            seq2_data = seq2_data.upper()
        
        # Write the data row to the output file
        # writer.writerow([protNm+' | '+ row[0], '5', classname, seq_data])

        # Write the data row to the output file
        writer.writerow(['5',  seq1_data+seq2_data])

        # Print the sequences
        print(f"Processing row {i} out of {total_rows}")
        
# Print the output file name
print(f"Sequences output file saved as: {output_file}")
