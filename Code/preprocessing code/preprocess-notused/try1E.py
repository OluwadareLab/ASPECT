import pybedtools
import numpy as np

# Function to one-hot encode a sequence
def one_hot_encode_sequence(sequence):
    nucleotides = ['A', 'C', 'G', 'T']
    encoding = np.zeros((len(sequence), 4), dtype=int)
    for i, nucleotide in enumerate(sequence):
        if nucleotide in nucleotides:
            encoding[i, nucleotides.index(nucleotide)] = 1
    return encoding

# Specify the path to the BED file
bed_file = 'filt_junc.bed'
fasta = 'hg38.fa'
# Specify the path to the output file
output_file = 'S140x2.tsv'
# Specify the path to save the numpy arrays
output_np_file = 'encoded_sequences.npy'

# Read the custom BED file into a BedTool object
bedtool = pybedtools.BedTool(bed_file)

# Create empty lists to store the encoded sequences
encoded_sequences = []

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
        seq1_data = seq1_file.read().strip()
        seq2_data = seq2_file.read().strip()

    # Remove the prefix ">chrY:57209910-57210050()" from the sequences
    seq1_data = seq1_data.replace(f">{interval[0]}:{start1}-{stop1}()", "")
    seq2_data = seq2_data.replace(f">{interval[0]}:{start2}-{stop2}()", "")

    # One-hot encode the sequences
    seq1_encoded = one_hot_encode_sequence(seq1_data)
    seq2_encoded = one_hot_encode_sequence(seq2_data)

    # Append the encoded sequences to the list
    encoded_sequences.append((seq1_encoded, seq2_encoded))

# Convert the list of encoded sequences to a numpy array
encoded_sequences_np = np.array(encoded_sequences)

# Save the numpy array to a .npy file
np.save(output_np_file, encoded_sequences_np)

# Open the output file in write mode
with open(output_file, 'w') as f:
    # Iterate over the encoded sequences
    for seq1_encoded, seq2_encoded in encoded_sequences:
        # Convert the encoded sequences to strings
        seq1_encoded_str = '\t'.join(map(str, seq1_encoded.ravel()))
        seq2_encoded_str = '\t'.join(map(str, seq2_encoded.ravel()))

        # Write the encoded sequences to the output file
        f.write(seq1_encoded_str + '\t' + seq2_encoded_str + '\n')

        # Print the encoded sequences
        print(seq1_encoded_str)
        print(seq2_encoded_str)
