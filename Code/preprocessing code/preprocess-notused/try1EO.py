import pybedtools
import numpy as np

# Function to one-hot encode a sequence
def one_hot_encode_sequence(sequence):
    nucleotides = ['A', 'C', 'G', 'T']
    encoding = np.zeros((len(sequence), 4), dtype=int)
    for i, nucleotide in enumerate(sequence):
        nucleotide = nucleotide.upper()
        if nucleotide in nucleotides:
            encoding[i, nucleotides.index(nucleotide)] = 1
    return encoding

# Specify the path to the BED file
bed_file = 'filt_junc.bed'
fasta = 'hg38.fa'
# Specify the path to the output file
output_file = 'S140x2.tsv'
# Specify the path to save the numpy array
output_np_file = 'Cons_OHE.npy'

# Read the custom BED file into a BedTool object
bedtool = pybedtools.BedTool(bed_file)

# Create an empty array to store the encoded sequences
encoded_sequences = np.zeros((len(bedtool), 280, 4), dtype=int)

# Iterate over each interval in the BedTool object
for i, interval in enumerate(bedtool):
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

    # One-hot encode the sequences
    seq1_encoded = one_hot_encode_sequence(seq1_data)
    seq2_encoded = one_hot_encode_sequence(seq2_data)

    # # Pad or truncate the sequences to length 140
    # seq1_encoded = seq1_encoded[:140]
    # seq2_encoded = seq2_encoded[:140]

    # Update the encoded_sequences array
    encoded_sequences[i, :seq1_encoded.shape[0]] = seq1_encoded
    encoded_sequences[i, :seq2_encoded.shape[0]] = seq2_encoded

# Save the encoded_sequences array to a .npy file
np.save(output_np_file, encoded_sequences, allow_pickle=False)

# Open the output file in write mode
with open(output_file, 'w') as f:
    # Iterate over the encoded sequences
    for i in range(encoded_sequences.shape[0]):
        # Convert the encoded sequences to strings
        seq1_encoded_str = ' '.join([str(x) for x in encoded_sequences[i, :, :].flatten()])
        seq2_encoded_str = ' '.join([str(x) for x in encoded_sequences[i, :, :].flatten(order='F')])

        # Write the encoded sequences to the output file
        f.write(seq1_encoded_str + '\t' + seq2_encoded_str + '\n')

        # Print the encoded sequences
        print(seq1_encoded_str)
        print(seq2_encoded_str)
