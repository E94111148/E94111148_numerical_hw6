# -*- coding: utf-8 -*-
"""
Created on Tue May  6 20:15:32 2025

@author: jerry
"""

import numpy as np

# 原始矩陣 A 和向量 b
A = np.array([
    [1.19, 2.11, -100, 1],
    [14.2, -0.112, 12.2, -1],
    [0, 100, -99.9, 1],
    [15.3, 0.110, -13.1, -1]
], dtype=float)

b = np.array([1.12, 3.44, 2.15, 4.16], dtype=float)

n = len(b)

# 組合成增廣矩陣
aug = np.hstack([A, b.reshape(-1, 1)])

# 高斯消去法 with pivoting
for i in range(n):
    # 找最大主元並交換列
    max_row = np.argmax(abs(aug[i:, i])) + i
    if max_row != i:
        aug[[i, max_row]] = aug[[max_row, i]]
    
    # 消去
    for j in range(i + 1, n):
        factor = aug[j][i] / aug[i][i]
        aug[j, i:] -= factor * aug[i, i:]

# 回代
x = np.zeros(n)
for i in range(n - 1, -1, -1):
    x[i] = (aug[i, -1] - np.dot(aug[i, i+1:n], x[i+1:n])) / aug[i, i]

# 輸出結果
print("解為：")
for i in range(n):
    print(f"x{i+1} = {x[i]:.6f}")
