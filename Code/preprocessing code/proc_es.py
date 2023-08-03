import numpy as np
import pybedtools
import csv
### Pre processes cassette exons

# Load data from the file
data = np.genfromtxt('../datasets/cassette100f.txt', delimiter='\t', dtype=object, skip_header=1, usecols=range(13), encoding='cp1252')

#Used to set up column after theyve been shuffled
start_ind = 2  # Index of the start column
end_ind = 3  # Index of the end column
count_ind = 4  # Index of the count column
skip_count = 8
incl_ind = 10
usage_ind3 = 11
usage_ind5 = 12

filtered_data = []
column_order = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11]

for i in range(len(data) - 1):  # Iterate until the second-to-last row
    pSkip = 1
    nSkip = 1
    current_row = data[i]
    start = int(current_row[start_ind].decode('cp1252'))  # Access start column of current row
    end = int(current_row[end_ind].decode('cp1252'))  # Access end column of current row

    if (end - start) >= 25:
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

# print(filtered_data[20], filtered_data[2000], filtered_data[50000], filtered_data[62858])
print("FILTDATA20: "+filtered_data[0])
breakpoint()
# Specify the path to the output CSV file
output_file = 'SEQ_ES.csv'
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
        start = int(row[1])
        stop = int(row[2])
        strand = row[3]
        skip_count = row[8]
        protNm = row[14]
        classL = [[2, 'Skipped Cassette'], [3, 'Non-Skipped Cassette']]
        i+=1

        # Create a BedTool object for the interval
        interval = pybedtools.BedTool(f"{row[0]}\t{start}\t{stop}\t{strand}", from_string=True)

        # Extract the sequence for the interval
        seq = interval.sequence(fi=fasta, s=True)

        # Read the sequence from the generated sequence file
        with open(seq.seqfn) as seq_file:
            seq_data = seq_file.read().strip().splitlines()[1]  # Skip the header line
            seq_data = seq_data.upper()
        
        # # Write the data row to the output file
        # if int(skip_count) >= 20:
        #     writer.writerow([protNm+' | '+ row[0] + ' | ' + skip_count, classL[0][0], classL[0][1], seq_data])
        # else:
        #     writer.writerow([protNm+' | '+ row[0] + ' | ' + skip_count, classL[1][0], classL[1][1], seq_data])

        # Write the data row to the output file
        if int(skip_count) >= 20:
            writer.writerow([classL[0][0], seq_data])
        else:
            writer.writerow([classL[1][0], seq_data])


        # Print the sequences
        print(f"Processing row {i} out of {total_rows}")
        
# Print the output file name
print(f"Sequences output file saved as: {output_file}")