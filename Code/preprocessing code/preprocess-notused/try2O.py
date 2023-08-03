import pybedtools

# Specify the path to the BED file
bed_file = 'filt_junc.bed'
fasta = 'hg38.fa'
# Specify the path to the output TSV file
output_file = 'sseeqq.tsv'

# Read the custom BED file into a BedTool object
bedtool = pybedtools.BedTool(bed_file)

# Open the output TSV file in append mode
with open(output_file, 'a') as f:

    # Iterate over each interval in the BedTool object
    for interval in bedtool:
        # Extract the start and stop positions for each interval
        start1 = int(interval[1])
        stop1 = int(interval[2])
        start2 = int(interval[3])
        stop2 = int(interval[4])
        strand = interval[5]

        # Create separate BedTool objects for each interval
        interval1 = pybedtools.BedTool(f"{interval[0]}\t{start1}\t{stop1}", from_string=True)
        interval2 = pybedtools.BedTool(f"{interval[0]}\t{start2}\t{stop2}", from_string=True)

        # Extract sequences for each interval
        seq1 = interval1.sequence(fi=fasta, s=True)
        seq2 = interval2.sequence(fi=fasta, s=True)

        # Read the sequences from the generated sequence files
        with open(seq1.seqfn) as seq1_file, open(seq2.seqfn) as seq2_file:
            seq1_data = seq1_file.read().strip()
            seq2_data = seq2_file.read().strip()

        # Write the interval information and sequences to the output TSV file
        # f.write(f"{interval[0]}\t{interval[1]}\t{interval[2]}\t{seq1_data}\t{interval[3]}\t{interval[4]}\t{seq2_data}\n")
        
        # Write the interval information and sequences to the output TSV file
        f.write(f"{seq1_data}\t{seq2_data}\n")
