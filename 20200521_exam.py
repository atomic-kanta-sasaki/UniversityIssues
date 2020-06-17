# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

data = np.array( [[-0.4000, -0.3333], [-0.5667, -0.2333], [0.2333, 0.2667], [0.0000, 0.2333], [0.0667, 0.3000], [-0.3333, -0.0667], [0.3333, 0.4667], [0.5000, 0.4333], [-0.6000, -0.3000], [-0.2333, 0.0667]])

x = np.array([-0.4, -0.5667, 0.2333, 0, 0.0667, -0.3333, 0.3333, 0.5, -0.6, -0.2333])
y = np.array([[-0.3333], [-0.2333], [0.2667], [0.2333], [0.3], [-0.0667], [0.4667], [0.4333], [-0.3], [0.0667]])

Phi = np.array([x ** i for i in range(6)])
theta = np.dot(np.dot(np.linalg.inv(np.dot(Phi, Phi.T)), Phi), y)
print('theta =', theta)

plt.scatter(x, y, color='blue')
xx = np.arange(x.min(), x.max(), (x.max() - x.min()) / 100.0)
P = np.array([xx ** i for i in range(6)])
plt.plot(xx, np.dot(P.T, theta), color='red', linewidth='1.0')
plt.show() 