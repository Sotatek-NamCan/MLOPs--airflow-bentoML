import pandas as pd
df = pd.read_csv("bank.csv", sep=';')  # nếu cần
print(df.columns.tolist())