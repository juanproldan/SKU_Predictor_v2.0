import pandas as pd
import os

try:
    # Change to the correct directory
    os.chdir(r'c:\Users\juanp\Documents\Python\0_Training\017_Fixacar\010_SKU_Predictor_v2.0')
    print(f"Current directory: {os.getcwd()}")

    # Check if file exists
    file_path = 'data/Maestro.xlsx'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit(1)

    print(f"Reading file: {file_path}")

    # Read the Excel file
    df = pd.read_excel(file_path)

    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if 'VIN_Year_Min' in df.columns and 'VIN_Year_Max' in df.columns:
        print("\nFirst 10 rows of year columns:")
        print(df[['VIN_Year_Min', 'VIN_Year_Max']].head(10))

        print("\nData types:")
        print(df[['VIN_Year_Min', 'VIN_Year_Max']].dtypes)

        print("\nUnique values in VIN_Year_Min:")
        print(df['VIN_Year_Min'].unique())

        print("\nUnique values in VIN_Year_Max:")
        print(df['VIN_Year_Max'].unique())

        print("\nAny null values?")
        print(f"VIN_Year_Min nulls: {df['VIN_Year_Min'].isnull().sum()}")
        print(f"VIN_Year_Max nulls: {df['VIN_Year_Max'].isnull().sum()}")
    else:
        print("Year columns not found in the file")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
