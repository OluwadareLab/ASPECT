import numpy as np

file_path = './low inclusion/x_three_data.npy'

loaded_data = np.load(file_path)

print("Shape of the loaded data:", loaded_data.shape)

dna = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}

with open("decoded_3data.csv", "w") as f:
        f.write('CLASS,SEQ\n')
        
for row in range(loaded_data.shape[0]):  # Iterate through rows
    seq = []
    for col in range(loaded_data.shape[1]-1):  # Iterate through columns
        ohe = []
        for element in range(loaded_data.shape[2]-1):  # Iterate through elements in each vector
            value = loaded_data[row, col, element]
            ohe.append(value)
            
        if ohe == dna['A']:
            seq.append('A')
        elif ohe == dna['C']:
            seq.append('C')
        elif ohe == dna['G']:
            seq.append('G')
        elif ohe == dna['T']:
            seq.append('T')
            
    with open("decoded_3data.csv", "a") as f:
        f.write('1,')
        for x in seq:
            f.write(str(x))
        f.write('\n')