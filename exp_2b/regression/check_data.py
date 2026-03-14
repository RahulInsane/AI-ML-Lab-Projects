import pandas as pd

# change filename here if needed
file_path = "regression/dataset/stock_data.xlsx"

df = pd.read_excel(file_path)

print("✅ Columns in dataset:")
print(df.columns)

print("\n✅ First 5 rows:")
print(df.head())
