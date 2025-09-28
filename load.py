import pandas as pd
real_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")
fake_df['label'] = 0
real_df['label'] = 1
data = pd.concat([real_df, fake_df], ignore_index=True)
print(data.head())
print(data.shape)
print(data.isnull().sum())
print(data['label'].value_counts())
