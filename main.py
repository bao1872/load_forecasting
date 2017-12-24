#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:20:14 2017

@author: zhenbao
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from itertools import islice
import xgboost as xgb

def window(seq, n=2):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
        
#读取原始数据
df = pd.read_excel('data.xlsx')

#选择有用的特征数据
df_load = df.DEMAND.values

#制作训练数据
X = np.array(list(window(df_load, n=int(24*1))))[:,:-1]
y = df_load[int(24*1)-1:]

#按照2：1的比例把训练数据划分为训练集和验证集数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#建立决策树模型
model = xgb.XGBRegressor()

#开始训练模型
model.fit(X_train, y_train)

#计算验证用的预测数据
y_pred = model.predict(X_test)

idx_start = 100
idx_end = 200

#可视化验证数据
plt.plot(y_test[idx_start:idx_end], color='b',alpha=0.3)
plt.plot(y_pred[idx_start:idx_end], '--', color='r',alpha=0.3)
plt.show()


