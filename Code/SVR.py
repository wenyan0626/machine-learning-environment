import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. 读取 Excel 文件
file_path = r"C:\Users\Desktop\Data.xlsx"
df = pd.read_excel(file_path)

# 2. 提取输入特征和目标输出
X = df.iloc[:, 2:17].values
y = df.iloc[:, 17:23].values

# 3. 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 特征标准化
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

# 定义矿物名称字典
mineral_names = {
    18: 'Ferrihydrite(Fh)%',
    19: 'Lepidocrocite(Lp)%',
    20: 'Goethite(Gt)%',
    21: 'Magnetite(Mt)%',
    22: 'Hematite(Hm)%'
}

# 用于存储结果的列表
results = []

# 5. 建立五个 SVR 模型，分别预测每列的数据
for i in range(5):  # 假设你确实有五列目标输出
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr.fit(X_train, y_train[:, i])  # 训练第 i 列的模型

    # 计算训练集和测试集的 R²、MSE 和 RMSE
    y_train_pred = svr.predict(X_train)
    train_r2 = round(r2_score(y_train[:, i], y_train_pred), 5)
    train_mse = round(mean_squared_error(y_train[:, i], y_train_pred), 5)
    train_rmse = round(np.sqrt(train_mse), 5)

    y_test_pred = svr.predict(X_test)
    test_r2 = round(r2_score(y_test[:, i], y_test_pred), 5)
    test_mse = round(mean_squared_error(y_test[:, i], y_test_pred), 5)
    test_rmse = round(np.sqrt(test_mse), 5)

    # 将结果添加到列表
    results.append([mineral_names[18 + i], train_r2, train_mse, train_rmse, test_r2, test_mse, test_rmse])

    # 打印结果
    print(f'{mineral_names[18 + i]}:')
    print(f'  Train R²: {train_r2:.5f}')
    print(f'  Train MSE: {train_mse:.5f}')
    print(f'  Train RMSE: {train_rmse:.5f}')
    print(f'  Test R²: {test_r2:.5f}')
    print(f'  Test MSE: {test_mse:.5f}')
    print(f'  Test RMSE: {test_rmse:.5f}')

# 将结果转换为 DataFrame
results_df = pd.DataFrame(results, columns=['Mineral', 'Train R²', 'Train MSE', 'Train RMSE', 'Test R²', 'Test MSE', 'Test RMSE'])

# 导出结果为 Excel 文件
results_file_path = r'D\RF\svr_results.xlsx'
results_df.to_excel(results_file_path, index=False)

print(f'结果已导出到 {results_file_path}')
