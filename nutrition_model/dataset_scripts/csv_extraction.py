import pandas as pd
import pdfplumber
import tabula
import camelot
import os
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PDFTableExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.output_dir = "extracted_tables"
        self.tables = []
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
    
    def extract_with_pdfplumber(self):
        """Extract tables using pdfplumber - good for simple tables"""
        print("Extracting tables with pdfplumber...")
        tables = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract tables from the page
                    page_tables = page.extract_tables()
                    
                    for table_num, table in enumerate(page_tables, 1):
                        if table and len(table) > 1:  # Ensure table has content
                            df = pd.DataFrame(table[1:], columns=table[0])
                            
                            # Clean the dataframe
                            df = self.clean_dataframe(df)
                            
                            if not df.empty:
                                table_info = {
                                    'name': f'pdfplumber_page_{page_num}_table_{table_num}',
                                    'data': df,
                                    'page': page_num,
                                    'method': 'pdfplumber'
                                }
                                tables.append(table_info)
                                print(f"Found table: {table_info['name']} ({df.shape[0]} rows, {df.shape[1]} cols)")
        
        except Exception as e:
            print(f"Error with pdfplumber: {e}")
        
        return tables
    
    def extract_with_tabula(self):
        """Extract tables using tabula-py - good for complex tables"""
        print("Extracting tables with tabula...")
        tables = []
        
        try:
            # Read all tables from PDF
            dfs = tabula.read_pdf(self.pdf_path, pages='all', multiple_tables=True)
            
            for i, df in enumerate(dfs, 1):
                if not df.empty:
                    # Clean the dataframe
                    df = self.clean_dataframe(df)
                    
                    if not df.empty:
                        table_info = {
                            'name': f'tabula_table_{i}',
                            'data': df,
                            'page': 'unknown',
                            'method': 'tabula'
                        }
                        tables.append(table_info)
                        print(f"Found table: {table_info['name']} ({df.shape[0]} rows, {df.shape[1]} cols)")
        
        except Exception as e:
            print(f"Error with tabula: {e}")
        
        return tables
    
    def extract_with_camelot(self):
        """Extract tables using camelot - good for PDF tables with borders"""
        print("Extracting tables with camelot...")
        tables = []
        
        try:
            # Try lattice method first (for tables with borders)
            try:
                camelot_tables = camelot.read_pdf(self.pdf_path, pages='all', flavor='lattice')
                method_suffix = 'lattice'
            except:
                # Fall back to stream method (for tables without borders)
                camelot_tables = camelot.read_pdf(self.pdf_path, pages='all', flavor='stream')
                method_suffix = 'stream'
            
            for i, table in enumerate(camelot_tables, 1):
                df = table.df
                
                if not df.empty:
                    # Clean the dataframe
                    df = self.clean_dataframe(df)
                    
                    if not df.empty:
                        table_info = {
                            'name': f'camelot_{method_suffix}_table_{i}',
                            'data': df,
                            'page': table.page,
                            'method': f'camelot_{method_suffix}'
                        }
                        tables.append(table_info)
                        print(f"Found table: {table_info['name']} ({df.shape[0]} rows, {df.shape[1]} cols)")
        
        except Exception as e:
            print(f"Error with camelot: {e}")
        
        return tables
    
    def clean_dataframe(self, df):
        """Clean and preprocess the dataframe"""
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Clean column names
        if not df.empty:
            df.columns = [self.clean_column_name(str(col)) for col in df.columns]
        
        # Remove rows that are mostly empty (more than 80% NaN)
        if not df.empty:
            threshold = len(df.columns) * 0.2  # Keep rows with at least 20% non-NaN values
            df = df.dropna(thresh=threshold)
        
        return df
    
    def clean_column_name(self, col_name):
        """Clean column names"""
        # Remove extra whitespace and newlines
        col_name = re.sub(r'\s+', ' ', str(col_name)).strip()
        
        # If column name is empty or just numbers, give it a generic name
        if not col_name or col_name.isspace() or col_name == 'nan':
            return 'Column'
        
        return col_name
    
    def extract_all_methods(self):
        """Extract tables using all available methods"""
        all_tables = []
        
        # Try each extraction method
        methods = [
            self.extract_with_pdfplumber,
            self.extract_with_tabula,
            self.extract_with_camelot
        ]
        
        for method in methods:
            try:
                tables = method()
                all_tables.extend(tables)
            except Exception as e:
                print(f"Method failed: {e}")
                continue
        
        self.tables = all_tables
        return all_tables
    
    def save_tables_to_csv(self):
        """Save all extracted tables to CSV files"""
        if not self.tables:
            print("No tables found to save.")
            return
        
        saved_files = []
        
        for table_info in self.tables:
            try:
                # Create filename
                filename = f"{table_info['name']}.csv"
                filepath = os.path.join(self.output_dir, filename)
                
                # Save to CSV
                table_info['data'].to_csv(filepath, index=False)
                saved_files.append(filepath)
                
                print(f"Saved: {filepath}")
                print(f"  - Shape: {table_info['data'].shape}")
                print(f"  - Method: {table_info['method']}")
                print(f"  - Page: {table_info['page']}")
                print()
                
            except Exception as e:
                print(f"Error saving {table_info['name']}: {e}")
        
        return saved_files
    
    def get_table_summary(self):
        """Get a summary of all extracted tables"""
        if not self.tables:
            return "No tables extracted."
        
        summary = f"Extracted {len(self.tables)} tables:\n\n"
        
        for i, table_info in enumerate(self.tables, 1):
            df = table_info['data']
            summary += f"{i}. {table_info['name']}\n"
            summary += f"   - Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            summary += f"   - Method: {table_info['method']}\n"
            summary += f"   - Page: {table_info['page']}\n"
            summary += f"   - Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}\n\n"
        
        return summary

def main():
    # Path to your PDF file
    pdf_path = "beef-cattle-only-tables.pdf"
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        print("Please make sure the file is in the same directory as this script.")
        return
    
    print(f"Processing PDF: {pdf_path}")
    print("="*50)
    
    # Create extractor instance
    extractor = PDFTableExtractor(pdf_path)
    
    # Extract tables using all methods
    tables = extractor.extract_all_methods()
    
    if tables:
        print("\n" + "="*50)
        print("EXTRACTION SUMMARY")
        print("="*50)
        print(extractor.get_table_summary())
        
        # Save tables to CSV
        print("="*50)
        print("SAVING TO CSV FILES")
        print("="*50)
        saved_files = extractor.save_tables_to_csv()
        
        print(f"\nTotal files saved: {len(saved_files)}")
        print(f"Output directory: {extractor.output_dir}")
        
    else:
        print("\nNo tables were extracted from the PDF.")
        print("This could mean:")
        print("1. The PDF contains images instead of text-based tables")
        print("2. The table structure is too complex")
        print("3. The PDF needs OCR processing")

# Additional utility functions
def install_requirements():
    """Print installation instructions for required packages"""
    requirements = """
To run this script, install the required packages:

pip install pandas pdfplumber tabula-py camelot-py[cv] openpyxl

Note: camelot-py might require additional system dependencies:
- For Ubuntu/Debian: sudo apt-get install python3-tk ghostscript
- For macOS: brew install ghostscript tcl-tk
- For Windows: Install ghostscript from https://www.ghostscript.com/download/gsdnld.html

Alternative lightweight version (if camelot installation fails):
pip install pandas pdfplumber tabula-py
"""
    print(requirements)

def extract_specific_table_patterns():
    """Extract tables with specific patterns for beef cattle nutrition"""
    # This function can be customized based on the specific table patterns
    # found in beef cattle nutrition documents
    pass

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("\nPlease install required packages first:")
        install_requirements()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nIf you're having issues with package installation,")
        print("try the lightweight version with just pdfplumber and tabula-py")