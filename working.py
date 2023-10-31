import pandas as pd

input_df = pd.read_csv("/home/jupyter-azureuser/hmeq.csv")
input_df


input_df.head(20)

input_df[input_df['REASON'] == 'DebtCon']

output_df = [input_df.head(20), input_df[input_df['REASON'] == 'DebtCon']]
output_df

input_df.head(20)

help(pd.merge)

len(output_df)