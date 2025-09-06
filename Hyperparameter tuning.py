import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. 读取Excel文件
file_path = r"C:\Users\Desktop\Data.xlsx"
df = pd.read_excel(file_path)

# 2. 提取输入特征和目标输出
X = df.iloc[:, 2:17].values
y = df.iloc[:, 17:23].values

# 3. 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# 4. 定义超参数网格
param_grid = {
    'n_estimators': [100, 200, 300, 500],  # 森林中树的数量
    'max_depth': [None, 10, 20, 30, 40],  # 树的最大深度
    'min_samples_split': [2, 5, 10],  # 内部节点再划分所需最小样本数
    'min_samples_leaf': [1, 2, 4],  # 叶子节点所需最小样本数
    'max_features': ['sqrt', 'log2', None]  # 分裂时考虑的最大特征数
}

# 5. 建立五个随机森林模型，分别预测每列的数据
for i in range(5):  # 假设你确实有五列目标输出
    rf = RandomForestRegressor(random_state=42)

    # 使用GridSearchCV进行参数调优
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2', verbose=2)
    grid_search.fit(X_train, y_train[:, i])

    # 获取最佳超参数的模型
    best_rf = grid_search.best_estimator_

    # 计算训练集和测试集的 R²、MSE 和 RMSE
    y_train_pred = best_rf.predict(X_train)
    train_r2 = round(r2_score(y_train[:, i], y_train_pred), 5)
    train_mse = round(mean_squared_error(y_train[:, i], y_train_pred), 5)
    train_rmse = round(np.sqrt(train_mse), 5)

    y_test_pred = best_rf.predict(X_test)
    test_r2 = round(r2_score(y_test[:, i], y_test_pred), 5)
    test_mse = round(mean_squared_error(y_test[:, i], y_test_pred), 5)
    test_rmse = round(np.sqrt(test_mse), 5)

    # 将结果添加到列表
    results.append([mineral_names[18 + i], train_r2, train_mse, train_rmse, test_r2, test_mse, test_rmse])

    # 打印结果
    print(f'{mineral_names[18 + i]}:')
    print(f'  Best Parameters: {grid_search.best_params_}')
    print(f'  Train R²: {train_r2:.5f}')
    print(f'  Train MSE: {train_mse:.5f}')
    print(f'  Train RMSE: {train_rmse:.5f}')
    print(f'  Test R²: {test_r2:.5f}')
    print(f'  Test MSE: {test_mse:.5f}')
    print(f'  Test RMSE: {test_rmse:.5f}')

# 将结果转换为DataFrame，并包含R²和RMSE
results_df = pd.DataFrame(results, columns=['Mineral', 'Train R²', 'Train MSE', 'Train RMSE', 'Test R²', 'Test MSE',
                                            'Test RMSE'])

# 导出结果为Excel文件
results_file_path = r'D\RF\random_forest_optimized_results.xlsx'
results_df.to_excel(results_file_path, index=False)

print(f'结果已导出到 {results_file_path}')
