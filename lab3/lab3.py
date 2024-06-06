import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier

# 读取数据
# 加载数据集
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# 预处理数据
data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
data['Embarked'].fillna('S', inplace=True)
data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])

# 填充缺失值
imputer = SimpleImputer(strategy='median')
data[['Age', 'Fare']] = imputer.fit_transform(data[['Age', 'Fare']])

# 特征缩放
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()
scaler_normalize = Normalizer()

data_standard = data.copy()
data_minmax = data.copy()
data_normalize = data.copy()

data_standard[['Age', 'Fare']] = scaler_standard.fit_transform(data_standard[['Age', 'Fare']])
data_minmax[['Age', 'Fare']] = scaler_minmax.fit_transform(data_minmax[['Age', 'Fare']])
data_normalize[['Age', 'Fare']] = scaler_normalize.fit_transform(data_normalize[['Age', 'Fare']])

# 异常值处理
iso_forest = IsolationForest(contamination=0.1)
outliers = iso_forest.fit_predict(data[['Age', 'Fare']])
data_no_outliers = data[outliers != -1]

# 替换异常值
data_imputed = data.copy()
data_imputed[['Age', 'Fare']] = SimpleImputer(strategy='median').fit_transform(data_imputed[['Age', 'Fare']])

# 处理非标准特征（已在预处理步骤中编码'Sex'和'Embarked'）

# 特征选择
X = data.drop('Survived', axis=1)
y = data['Survived']

# 过滤方法
selector_var = VarianceThreshold(threshold=0.1)
X_filtered = selector_var.fit_transform(X)

# 包裹方法
selector_rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
X_wrapped = selector_rfe.fit_transform(X, y)

# 嵌入方法
model_lasso = Lasso(alpha=0.1)
model_lasso.fit(X, y)
importance = np.abs(model_lasso.coef_)
selected_features = X.columns[np.argsort(importance)[-5:]]

# 输出结果
print("Standard Scaled Data:\n", data_standard.head())
print("Min-Max Scaled Data:\n", data_minmax.head())
print("Normalized Data:\n", data_normalize.head())
print("Data without Outliers:\n", data_no_outliers.head())
print("Data with Imputed Values:\n", data_imputed.head())
print("Filtered Data Shape:\n", X_filtered.shape)
print("Wrapped Data Shape:\n", X_wrapped.shape)
print("Selected Features by Lasso:\n", selected_features)
