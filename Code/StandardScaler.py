import pandas as pd
from sklearn.preprocessing import StandardScaler

#  df：DataFrame，包含数值型特征
df = pd.read_csv('your_dataset.csv')

# 选择需要标准化的特征列
features = ['feature1', 'feature2', 'feature3']  # 根据实际情况修改

# 初始化 StandardScaler
scaler = StandardScaler()

# 对选定特征进行标准化
df[features] = scaler.fit_transform(df[features])

# 查看标准化后的数据
print(df.head())