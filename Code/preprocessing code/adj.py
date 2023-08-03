import pandas as pd
###########
#Takes data file and adjusts the labels to start at 1,2 
##########
def modify_class_column(input_file, target_value):
    data = pd.read_csv(input_file)

    # Check if the "CLASS" column equals the target_value
    mask = data["CLASS"] == int(target_value)

    # Modify the "CLASS" column for rows that match the condition
    data.loc[mask, "CLASS"] = 2

    # Save the modified data to a new CSV file or overwrite the original file
    data.to_csv(input_file, index=False)

if __name__ == "__main__":
    input_file = "../tNt/train_est.csv"  # Replace with the path to your input file

    # Call the function to modify the "CLASS" column
    modify_class_column(input_file, '4')

    # input_file = "./backwards/trunct_3data.csv"  # Replace with the path to your input file

    # # Call the function to modify the "CLASS" column
    # modify_class_column(input_file, '1')
