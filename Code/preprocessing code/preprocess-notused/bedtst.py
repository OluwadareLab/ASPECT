import pybedtools

# Specify the path to the input TSV file
input_file = 'fin_out.tsv'

# Specify the path to the output BED file
output_file = 'my.bed'

# Specify the column indices to extract (1-based index)
column_indices = [0, 1, 2, 3, 20]

# Read the TSV file into a BedTool object
bedtool = pybedtools.BedTool(input_file)

# Convert the TSV file to a BED file with the specified columns
bed6 = bedtool.cut(column_indices).saveas(output_file)

# Print the converted BED file
print(f"Converted BED file saved as: {output_file}")
