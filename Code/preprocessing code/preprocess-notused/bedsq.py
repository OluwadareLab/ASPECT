import pandas as pd
import pybedtools
from pybedtools import BedTool

myBedTool = BedTool('filt_junc.bed')


# Read the TSV file into a pandas DataFrame
data = pd.read_csv('filt_data.tsv', delimiter='\t', header=None)

# Initialize an empty list to store the extracted sequences
extracted_sequences = []

# Iterate over the rows in the DataFrame
for index, row in data.iterrows():
    chrm = row[0]
    strand = row[1]
    l_start = int(row[2])
    l_end = int(row[3])
    r_start = int(row[4])
    r_end = int(row[5])

    # Create BED intervals for sequence set 1
    bed_set_1 = pybedtools.BedTool(
        f"{chrm}\t{l_start}\t{l_end}\t{strand}", from_string=True
    )

    # Create BED intervals for sequence set 2
    bed_set_2 = pybedtools.BedTool(
        f"{chrm}\t{r_start}\t{r_end}\t{strand}", from_string=True
    )

    # Extract sequences from the FASTA file using modified BED intervals
    extracted_sequences = [
        bed_set_1.sequence(fi='hg38.fa', s=True),
        bed_set_2.sequence(fi='hg38.fa', s=True),
    ]

    with open('seqncd.tsv', 'a') as f:
        for seq in extracted_sequences:
            f.write(f'{chrm}\t{strand}\t{l_start}\t{l_end}\t{seq}\n')

