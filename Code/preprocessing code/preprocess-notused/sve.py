import csv

def rearrange_columns(input_file, output_file, column_order):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')
        for row in reader:
            new_row = []
            for column in column_order:
                new_row.append(row[column])
            writer.writerow(new_row)

if __name__ == '__main__':
    input_file = 'filt_data.tsv'
    output_file = 'filt_ard.tsv'
    column_order = [0, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20 ]
    rearrange_columns(input_file, output_file, column_order)
