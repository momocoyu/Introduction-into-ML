
import numpy as np
'''
import matplotlib.pyplot as plt
mean = [1, 1]
cov = [[1, -0.5], [-0.5, 1]]  # diagonal covariance
x1, x2 = np.random.multivariate_normal(mean, cov, 1000).T
plt.plot(x1, x2, 'x')
plt.axis('equal')
plt.show()
'''
a2=np.array([[1,0],[1,3]])   #建立一个二维数组

print(np.linalg.eig(a2) )               #返回矩阵a2的特征值与特征向量

