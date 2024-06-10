import pandas as pd
import openpyxl

# Load the Excel file with the correct header row
file_path = r'C:\Users\Lenovo\Desktop\Proj Work\TEST\RawTest.xlsx'
excel_data = pd.read_excel(file_path, header=0)

# Load the list of columns to keep from the text file
columns_to_keep_file = r'C:\Users\Lenovo\Desktop\Proj Work\TEST\list.txt'
with open(columns_to_keep_file, 'r') as file:
    columns_to_keep = [line.strip() for line in file.readlines()]

# Filter the columns
filtered_data = excel_data[columns_to_keep]

# Save the filtered data to a new Excel file
output_path = r'C:\Users\Lenovo\Desktop\Proj Work\TEST\filtered_data.xlsx'
filtered_data.to_excel(output_path, index=False)

output_path
