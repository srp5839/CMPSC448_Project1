
import pandas as pd
from pathlib import Path

# Find the folder containing your Python script
folder = Path(__file__).resolve().parent

# Load the CSV with Windows encoding
df = pd.read_csv(
    folder / "Project_448.csv",
    encoding="cp1252"
)

# Display the first five rows
print(df.head())

# Display the column names
print(df.columns.tolist())

# Count responses from each LLM
print(df["LLM_Name"].value_counts())

# Check for missing values
print(df.isnull().sum())

# Remove rows containing any NA values
df = df.dropna()

# Reset the row indexes
df = df.reset_index(drop=True)

# Check dataset dimensions
print(df.shape)

#Now we have cleaned the dataset by removing rows with missing values
