import deepchem as dc
import os

#数据加载
file = 'Data\\bbbp.csv'
featurizer = dc.feat.CircularFingerprint(size=128)
loader = dc.data.CSVLoader(["p_np"], feature_field="smiles", featurizer=featurizer)
# tasks:label的列明, featurizer：分子特征化对象
dataset = loader.create_dataset(sdf_file)
# print(dataset)
# print(dataset.X) #分子的特征
# print(dataset.y) #label,也就是tasks
# print(dataset.w) #样本权重，均为1
# print(dataset.ids)

# #划分数据集
# X = dataset.X
# Y = dataset.y
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# #sklearn模型
# reg = LinearRegression()
# reg.fit(x_train, y_train)
# score = reg.score(x_test, y_test)
# print('Simels线性回归模型的分数为:',score)

#划分数据集
import numpy as np
from deepchem import splits
splitter = splits.RandomSplitter()
train_set, test_set = splitter.train_test_split(dataset, frac_train=0.7)
#建立图神经卷积网络模型
model = dc.models.GraphConvModel(1, mode='regression', dropout=0.5)
model.fit(train_set) #会有警告
#评估器
avg_pearson_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
test_score = model.evaluate(test_set,[avg_pearson_r2])
print('图卷积网络模型的预测分数为:', test_score)