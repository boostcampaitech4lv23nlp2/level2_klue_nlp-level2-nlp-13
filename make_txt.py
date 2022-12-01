import csv
csv_file = './data/raw_data/test_data.csv'
txt_file = './data/preprocessed_data/test.txt'
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [my_output_file.write(row[1]+'\n') for row in csv.reader(my_input_file)]
    my_output_file.close()