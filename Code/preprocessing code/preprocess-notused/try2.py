import pybedtools

# Specify the path to the BED file
bed_file = 'filt_junc.bed'
fasta = 'hg38.fa'
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
    interval1 = pybedtools.BedTool(f"{interval[0]}\t{start1}\t{stop1}", from_string=True)
    interval2 = pybedtools.BedTool(f"{interval[0]}\t{start2}\t{stop2}", from_string=True)

    # Extract sequences for each interval
    seq1 = interval1.sequence(fi=fasta, s=True)
    seq2 = interval2.sequence(fi=fasta, s=True)

    # Process the extracted sequences as desired
    # ...

    # Print the sequences
    print(f"Sequence 1: {print(open(seq1.seqfn).read())}")
    print(f"Sequence 2: {print(open(seq2.seqfn).read())}")
