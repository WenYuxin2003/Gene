import pandas as pd

file_path = 'mcf7.csv'

# 读取 CSV 并显示前 5 行
df = pd.read_csv(file_path)

print("数据维度:", df.shape)
print(df.head())   # 显示前 5 行
