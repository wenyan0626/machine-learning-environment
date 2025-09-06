import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# 1. 读取Excel文件
file_path = r"C:\Users\Desktop\Data.xlsx"
df = pd.read_excel(file_path)

# 2. 提取输入特征和目标输出
X = df.iloc[:, 2:17].values
y = df.iloc[:, 17:23].values

# 3. 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape X 为 [样本数, 时间步, 特征数]，这里的时间步设置为 1，因为数据不是时间序列
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

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

# 4. 建立五个LSTM模型，分别预测每列的数据
for i in range(5):  # 假设你确实有五列目标输出
    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))  # 输出层，预测单一值

    model.compile(optimizer=Adam(), loss='mean_squared_error')

    # 训练模型
    model.fit(X_train, y_train[:, i], epochs=100, batch_size=32, verbose=0)

    # 预测训练集和测试集
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 计算训练集和测试集的 R²、MSE 和 RMSE
    train_r2 = round(r2_score(y_train[:, i], y_train_pred), 5)
    train_mse = round(mean_squared_error(y_train[:, i], y_train_pred), 5)
    train_rmse = round(np.sqrt(train_mse), 5)

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

# 将结果转换为DataFrame
results_df = pd.DataFrame(results, columns=['Mineral', 'Train R²', 'Train MSE', 'Train RMSE', 'Test R²', 'Test MSE', 'Test RMSE'])

# 导出结果为Excel文件
results_file_path = r'D\RF\lstm_results.xlsx'
results_df.to_excel(results_file_path, index=False)

print(f'结果已导出到 {results_file_path}')
