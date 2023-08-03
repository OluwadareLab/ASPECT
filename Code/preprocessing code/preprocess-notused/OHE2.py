import pybedtools
import numpy as np
import pandas as pd

# Specify the path to the BED file
bed_file = 'filt_junc.bed'
fasta = 'hg38.fa'
output_file = 'ohe_out2.tsv'  # Specify the path to the output file

# Define a dictionary for one-hot encoding
one_hot_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}

# Initialize a list to store one-hot encoded sequences
encoded_seqs = []

# Read the custom BED file into a BedTool object
bedtool = pybedtools.BedTool(bed_file)

# Iterate over each interval in the BedTool object
for interval in bedtool:
    # Extract the start and stop positions for each interval
    start1 = int(interval[1])
    stop1 = int(interval[2])
    start2 = int(interval[3])
    stop2 = int(interval[4])
    strand = interval[5]

    # Create separate BedTool objects for each interval
    interval1 = pybedtools.BedTool(f"{interval[0]}\t{start1}\t{stop1}\t.\t.\t{strand}", from_string=True)
    interval2 = pybedtools.BedTool(f"{interval[0]}\t{start2}\t{stop2}\t.\t.\t{strand}", from_string=True)

    # Extract sequences for each interval
    seq1 = interval1.sequence(fi=fasta, s=True)
    seq2 = interval2.sequence(fi=fasta, s=True)

    # Read the sequences from the generated sequence files
    with open(seq1.seqfn) as seq1_file, open(seq2.seqfn) as seq2_file:
        seq1_data = seq1_file.read().strip().splitlines()[1].upper()  # Skip the header line and convert to uppercase
        seq2_data = seq2_file.read().strip().splitlines()[1].upper()  # Skip the header line and convert to uppercase

    # One-hot encode the sequences
    encoded_seq1 = [one_hot_dict[base] for base in seq1_data if base in one_hot_dict]
    encoded_seq2 = [one_hot_dict[base] for base in seq2_data if base in one_hot_dict]

    # Pad the sequences with zeros to a length of 140
    encoded_seq1 = encoded_seq1 + [[0, 0, 0, 0]] * (140 - len(encoded_seq1))
    encoded_seq2 = encoded_seq2 + [[0, 0, 0, 0]] * (140 - len(encoded_seq2))

    # Combine the two sequences
    combined_seq = encoded_seq1 + encoded_seq2

    # Add the combined sequence to the list
    encoded_seqs.append(combined_seq)

# Convert the list to a numpy array
encoded_seqs = np.array(encoded_seqs)

# Save the numpy array to a NPY file
np.save('encoded_seqs.npy', encoded_seqs)

# Save the numpy array to a TSV file
with open(output_file, 'w') as f:
    for seq in encoded_seqs:
        f.write('\t'.join([str(x) for sublist in seq for x in sublist]) + '\n')
