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

costlog_ = []
costsq_ = []

for i in range(500):
    u1 = w1.dot(X)
    z1 = sigmoid(u1)
    u2 = w2.dot(z1)
    z2 = sigmoid(u2)
    delta2 = z2 - y
    dw2 = delta2.dot(z1.T)
    w2 -= dw2
    delta1 = w2.T.dot(delta2) * (z1 * (1 - z1))
    dw1 = delta1.dot(X.T)
    w1 -= dw1
    w2 -= dw2
    costlog = -np.sum(y * np.log(z2) + (1 - y) * np.log(1 - z2))
    costlog_.append(costlog)
    costsq = 0.5 * np.sum(delta2 ** 2)
    costsq_.append(costsq)

# plt.plot(costlog_)
# plt.plot(costsq_, linestyle="dashed")
# plt.show()
print(z2)