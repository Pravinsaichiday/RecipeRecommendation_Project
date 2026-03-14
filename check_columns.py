import pandas as pd

# --- CONFIGURATION ---
FILE_TO_CHECK = 'cuisine_updated.csv'
# ---------------------

try:
    # We only load the first 5 rows to make this fast
    df = pd.read_csv(FILE_TO_CHECK, nrows=5)
    
    print(f"\n--- Found these column names in {FILE_TO_CHECK} ---")
    print(df.columns.tolist())
    print("-------------------------------------------------\n")

except FileNotFoundError:
    print(f"\nERROR: The file '{FILE_TO_CHECK}' was not found.")
    print("Please make sure it's in the same folder as this script.\n")
except Exception as e:
    print(f"\nAn error occurred: {e}\n")