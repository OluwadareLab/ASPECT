import numpy as np

# Load data from the file
data = np.genfromtxt('result.tsv', delimiter='\t', dtype=object, skip_header=1, usecols=range(21), encoding='cp1252')

print('Data shape:', data.shape)

start_ind = 2  # Index of the start column
end_ind = 3  # Index of the end column
count_ind = 4  # Index of the count column

filtered_data = []

for i in range(len(data) - 1):  # Iterate until the second-to-last row
    pSkip = 1
    nSkip = 1
    current_row = data[i]
    count = int(current_row[count_ind].decode('cp1252'))  # Access count column of current row
    start = int(current_row[start_ind].decode('cp1252'))  # Access start column of current row
    end = int(current_row[end_ind].decode('cp1252'))  # Access end column of current row

    if count >= 20 and (end - start) >= 25:
        if i + 1 < len(data):
            nSkip = 0
            next_row = data[i + 1]
            next_start = int(next_row[start_ind].decode('cp1252'))  # Access start column of next row
        if i - 1 >= 0:
            pSkip = 0
            prev_row = data[i - 1]
            prev_end = int(prev_row[end_ind].decode('cp1252'))  # Access end column of next row

        if pSkip == 1 and (next_start - end) >= 80:
            # Prossesing when there is no row before
            new_values = [item.decode('cp1252') for item in current_row]
            new_values[5] = str(0)
            new_values[6] = str(end - start)
            new_values[7] = str(next_start - end)
            new_values[8] = str(start - 70)
            new_values[9] = str(start + 70)
            new_values[10] = str(end - 70)
            new_values[11] = str(end + 70)
            filtered_data.append(new_values)

        elif nSkip == 1 and (start - prev_end) >= 80:
            # Filtering when there is no row after
            new_values = [item.decode('cp1252') for item in current_row]
            new_values[5] = str(start - prev_end)
            new_values[6] = str(end - start)
            new_values[7] = str(0)
            new_values[8] = str(start - 70)
            new_values[9] = str(start + 70)
            new_values[10] = str(end - 70)
            new_values[11] = str(end + 70)
            filtered_data.append(new_values)

        elif (start - prev_end) >= 80 and (next_start - end) >= 80:
            new_values = [item.decode('cp1252') for item in current_row]
            new_values[5] = str(start - prev_end)
            new_values[6] = str(end - start)
            new_values[7] = str(next_start - end)
            new_values[8] = str(start - 70)
            new_values[9] = str(start + 70)
            new_values[10] = str(end - 70)
            new_values[11] = str(end + 70)
            filtered_data.append(new_values)

# Save the filtered data to a file
with open('filt_data.tsv', 'w') as f:
    for row in filtered_data:
        f.write('\t'.join(row) + '\n')