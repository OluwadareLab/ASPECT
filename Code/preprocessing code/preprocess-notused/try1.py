import pybedtools

# Specify the path to the BED file
bed_file = 'filt_junc.bed'
fasta = 'hg38.fa'
# Specify the path to the output file
output_file = 'S140x2.tsv'

# Read the custom BED file into a BedTool object
bedtool = pybedtools.BedTool(bed_file)

# Open the output file in write mode
with open(output_file, 'w') as f:
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
            seq1_data = seq1_file.read().strip().splitlines()[1]  # Skip the header line
            seq2_data = seq2_file.read().strip().splitlines()[1]  # Skip the header line

        # Write the sequences to the output file
        f.write(seq1_data + '\t' + seq2_data + '\n')

        # Print the sequences
        print(seq1_data)
        print(seq2_data)
