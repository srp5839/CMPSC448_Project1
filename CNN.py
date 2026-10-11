import pandas as pd 

from pathlib import Path

from sklearn.preprocessing import LabelEncoder

folder_path = Path(__file__).resolve().parent

df = pd.read_csv(folder_path / "Project_448.csv",
                 encoding = "cp1252")

df = df.dropna().reset_index(drop=True)

x = df["LLM_Output"]

y = df["LLM_Name"]

encoder = LabelEncoder()

y = encoder.fit_transform(y)

print('Total Responses:', len(x))

print("LLM Classes:", encoder.classes_)

print("First Encoder Label:", y[0])
