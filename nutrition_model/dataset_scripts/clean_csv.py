import pandas as pd
import numpy as np

def clean_csv_file():
    # Read the CSV file
    df = pd.read_csv('extracted_tables/tabula_table_4.csv')
    
    # Define the new column names
    new_columns = [
        'current_weight', 'current_bcs', 'mature_wt', 'birth_weight', 
        'tissue_adg', 'uterus_adg', 'dm_intake_lb_day', 'dm_intake_pct_bw',
        'dm_intake_total', 'dm_pct', 'nem_mcal_day', 'nem_mcal_lb',
        'crude_protein_lb_day', 'crude_protein_pct', 'calcium_g_day',
        'calcium_pct', 'phosphorous_g_day', 'phosphorous_pct'
    ]
    
    # Function to clean and split the first column
    def clean_first_column(row):
        # Remove quotes and split by spaces
        values = str(row).replace('"', '').split()
        # Extract numeric values
        numeric_values = []
        for val in values:
            try:
                numeric_values.append(float(val.replace(',', '')))
            except ValueError:
                continue
        return numeric_values[:6]  # Return first 6 numeric values
    
    # Function to clean other columns
    def clean_other_columns(row):
        if isinstance(row, str):
            return [float(x.strip()) for x in row.split() if x.strip() and x.strip() != 'TDN']
        return []

    # Process the data
    processed_data = []
    
    for _, row in df.iterrows():
        new_row = []
        # Process first column
        first_col_values = clean_first_column(row.iloc[0])
        new_row.extend(first_col_values)
        
        # Process other columns
        for col in df.columns[1:]:
            values = clean_other_columns(str(row[col]))
            new_row.extend(values)
            
        if len(new_row) == len(new_columns):  # Only add complete rows
            processed_data.append(new_row)
    
    # Create new dataframe
    clean_df = pd.DataFrame(processed_data, columns=new_columns)
    
    # Save to new CSV file
    clean_df.to_csv('extracted_tables/clean_table_2.csv', index=False)
    print("CSV file has been cleaned and saved as 'clean_table_2.csv'")

if __name__ == "__main__":
    clean_csv_file()