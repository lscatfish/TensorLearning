from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from mt.core import net, optm
from mt.core import Placeholder, Session
from mt.core.function import measure
from mt.core.util import numpy_one_hot


raw_data = np.asarray([
    # 密度  含糖率  好瓜
    [0.697, 0.460, 1],
    [0.774, 0.376, 1],
    [0.634, 0.264, 1],
    [0.608, 0.318, 1],
    [0.556, 0.215, 1],
    [0.403, 0.237, 1],
    [0.481, 0.149, 1],
    [0.437, 0.211, 1],
    [0.666, 0.091, 0],
    [0.243, 0.267, 0],
    [0.245, 0.057, 0],
    [0.343, 0.099, 0],
    [0.639, 0.161, 0],
    [0.567, 0.198, 0],
    [0.360, 0.370, 0],
    [0.593, 0.042, 0],
    [0.719, 0.103, 0],
])
"""这个数据不是很好，最后训练集的准确率应该在0.75左右"""

train_X = raw_data[:, :-1].copy()
train_Y = np.astype(raw_data[:, -1], np.int32)

train_X = (train_X - train_X.min(axis = 0)) / np.ptp(train_X, axis = 0)

label = numpy_one_hot(train_Y)
X = Placeholder()
Y = Placeholder()

"""
二分类，双特征
input(2) -> linear(2,5) -> Relu -> linear(5,2) -> Softmax -> output_p(2)
"""
# TODO :实际上，这里activate_func应该设计成回调函数，或者用net包装
out1 = net.Linear(2, 5, activate_func = "relu", init = 'randn')(X)
out2 = net.Linear(5, 2, activate_func = "softmax", init = 'randn')(out1)

loss = measure.CrossEntropy(reduction = "mean")(predict = out2, label = Y)
session = Session()
optimizer = optm.Adam(learning_rate = 0.1)  # 这个优化器比sgd随机梯度下降好多了

losses = []
acces = []
for epoch in range(50):
    session.run(root_op = loss, feed_dict = {X: train_X, Y: label})
    optimizer.backward(loss)
    pre_lab = np.argmax(out2.numpy, axis = 1)
    print(pre_lab)
    acc = accuracy_score(train_Y, pre_lab)
    print(f"\033[32m[Epoch:{epoch}]\033[0m loss: {loss.numpy} accuracy: {acc}")
    losses.append(loss.numpy)
    acces.append(acc)
    optimizer.zero_grad()

plt.plot(losses, label = "loss")
plt.plot(acces, label = "acc")
plt.legend()
plt.show()
