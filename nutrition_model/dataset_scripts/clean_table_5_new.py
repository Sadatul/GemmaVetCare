import csv
import re

def clean_row(row):
    # Split the first value which contains the body weight and ADG
    if row and row[0]:
        bw_adg = row[0].split()
        body_weight = bw_adg[0]
        adg = bw_adg[1] if len(bw_adg) > 1 else ""
    else:
        body_weight = ""
        adg = ""
    
    # Handle the rest of the row
    cleaned = []
    for value in row[1:]:
        if value:
            # Split values separated by spaces
            parts = value.strip().split()
            cleaned.extend(parts)
        else:
            cleaned.append("")
            
    return [body_weight, adg] + cleaned

# Read and process the file
input_file = "extracted_tables/tabula_table_13.csv"
output_file = "extracted_tables/growing_mature_bulls_2300_pounds_at_finishing.csv"

with open(input_file, 'r') as infile:
    reader = csv.reader(infile)
    rows = list(reader)

# Process each row
cleaned_rows = [clean_row(row) for row in rows]

# Define headers
headers = [
    "Body weight (lbs)",
    "ADG (lbs)",
    "DM Intake (lbs/day)",
    "TDN (% DM)",
    "NEm (Mcal/lb)",
    "NEg (Mcal/lb)",
    "CP (% DM)",
    "Ca (%DM)",
    "P (% DM)",
    "TDN (lbs)",
    "NEm (Mcal)",
    "NEg (Mcal)",
    "CP (lbs)",
    "Ca (grams)",
    "P (grams)"
]

# Write the headers and cleaned data
with open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(headers)
    writer.writerows(cleaned_rows)

print("File has been cleaned and saved as tabula_table_5_cleaned.csv")
