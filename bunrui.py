import numpy as np
import matplotlib.pyplot as plt

r1 = np.linspace(0.1, 10, 200)
theta1 = np.linspace(0, 6 * np.pi, 200)


X = [[r1 * np.cos(theta1), r1 * np.sin(theta1)], [r1 * np.cos(theta1 + np.pi), r1 * np.sin(theta1 + np.pi)]]

# plt.scatter(r1 * np.cos(theta1), r1 * np.sin(theta1), marker='.', color='sienna')
# plt.scatter(r1 * np.cos(theta1 + np.pi), r1 * np.sin(theta1 + np.pi), marker='.', color='orangered')
# plt.show()

hoge1 = r1 * np.cos(theta1)
hoge1 += (r1 * np.cos(theta1 + np.pi))
hoge2 = r1 * np.sin(theta1)
hoge2 += (r1 * np.sin(theta1 + np.pi))

j = []
y = []
for i in range(200):
    if i < 99:
        j.append(0)
    else:
        j.append(1)

y.insert(-1, j)
K = []
for i in range(200):
    K.insert(-1, [hoge1[i], hoge2[i]])

# print(K)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.array(K).T

m = X.shape[1]
# y = np.array([[0, 1, 1, 0]])
one = np.ones([1, m])
X = np.vstack([one, X])
hidden = 5

w1 = np.random.rand(hidden, X.shape[0])

w2 = np.random.rand(1, hidden)

costlog_ = []
costsq_ = []

for i in range(500):
    u1 = w1.dot(X)
    # print(u1)
    z1 = sigmoid(u1)
    # print(z1)
    u2 = w2.dot(z1)
    # print(u2)
    z2 = sigmoid(u2)
    # print(z2)
    delta2 = z2 - y
    dw2 = delta2.dot(z1.T)
    w2 -= dw2
    delta1 = w2.T.dot(delta2) * (z1 * (1 - z1))
    dw1 = delta1.dot(X.T)
    w1 -= dw1
    # 評価関数
    z2 = np.array(z2)
    y = np.array(y)
    # クロスエントロピー
    costlog = -np.sum(y * np.log(z2) + (1 - y) * np.log(1 - z2))
    costlog_.append(costlog)
    # 二乗誤差
    costsq = 0.5 * np.sum(delta2 ** 2)
    costsq_.append(costsq)
    print(costsq_)
    

plt.plot(costlog_)
plt.plot(costsq_, linestyle="dashed")
plt.show()
# print(z2)