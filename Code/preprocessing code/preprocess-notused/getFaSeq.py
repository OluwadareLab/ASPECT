import pybedtools

# Specify the path to the BED file
bed_file = 'my.bed'
fasta = 'hg38.fa'
output_file = 'sequences_output.csv'  # Specify the path to the output file

# Count the total number of rows in the BED file
total_rows = sum(1 for _ in open(bed_file))
i = 1
# Open the output file in write mode
with open(output_file, 'w') as f:
    # Read the custom BED file into a BedTool object
    bedtool = pybedtools.BedTool(bed_file)

    # Iterate over each interval in the BedTool object
    for interval in bedtool:
        # Extract the start and stop positions for each interval
        start = int(interval[1])
        stop = int(interval[2])
        strand = interval[3]
        protNm = interval[4]
        classname = "Constitutive Exon"
        i+=1

        # Create separate BedTool objects for each interval
        interval1 = pybedtools.BedTool(f"{interval[0]}\t{start}\t{stop}\t{strand}", from_string=True)


        # Extract sequences for each interval
        seq1 = interval1.sequence(fi=fasta, s=True)

        # Read the sequences from the generated sequence files
        with open(seq1.seqfn) as seq1_file:
            seq1_data = seq1_file.read().strip().splitlines()[1]  # Skip the header line
            seq1_data = seq1_data.upper()
            
        # Write the sequences to the output file
        f.write(protNm + ',' + '1' + ',' + classname + ',' + seq1_data  + '\n')

        # Print the sequences
        print(f"Processing row {i} out of {total_rows}")
