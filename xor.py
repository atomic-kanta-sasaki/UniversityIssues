import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
m = X.shape[1]
y = np.array([[0, 1, 1, 0]])
one = np.ones([1, m])
X = np.vstack([one, X])

hidden = 3

w1 = np.random.rand(hidden, X.shape[0])

w2 = np.random.rand(1, hidden)
nizyou_gosa_ = []


for i in range(500):
    u1 = w1.dot(X)
    print(u1)
    z1 = sigmoid(u1)
    u2 = w2.dot(z1)
    
    z2 = sigmoid(u2)
    delta2 = z2 - y
    dw2 = delta2.dot(z1.T)
    w2 -= dw2
    delta1 = w2.T.dot(delta2) * (z1 * (1 - z1))
    dw1 = delta1.dot(X.T)
    w1 -= dw1
    
    
    nizyou_gosa = 0.5 * np.sum(delta2 ** 2)
    nizyou_gosa_.append(nizyou_gosa)

plt.plot(nizyou_gosa_)
plt.show()
# print(z2)