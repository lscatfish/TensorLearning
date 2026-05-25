"""
鸢尾花分类 — mt 自研框架神经网络（独立版本）

使用自研 mt 框架实现全连接神经网络对 Iris 数据集进行分类。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import mt.core.net
import mt.core.optm
from mt.core.base import Placeholder, Session
from mt.core.function import measure
from mt.core.util import numpy_one_hot

# ============================================================
# 1. 数据加载与预处理
# ============================================================
print("=" * 60)
print("1. 加载 Iris 数据集")
print("=" * 60)

DATA_PATH = os.path.join(os.path.dirname(__file__), "iris.data")
data = np.loadtxt(DATA_PATH, delimiter=",", dtype=str)

X_raw = data[:, :-1].astype(np.float64)          # 4 个特征
y_raw = data[:, -1]                               # 类别标签字符串

class_names = np.unique(y_raw)
class_to_idx = {c: i for i, c in enumerate(class_names)}
y_labels = np.array([class_to_idx[c] for c in y_raw])

print(f"  样本数: {X_raw.shape[0]}")
print(f"  特征数: {X_raw.shape[1]} (花萼长、花萼宽、花瓣长、花瓣宽)")
print(f"  类别数: {len(class_names)} — {list(class_names)}")
print(f"  各类别数量: {[np.sum(y_labels == i) for i in range(3)]}")

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_labels, test_size=0.3, random_state=42, stratify=y_labels
)

# 标准化 — 对 NN 很重要
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")

# ============================================================
# 2. mt 自研框架神经网络
# ============================================================
print("\n" + "=" * 60)
print("2. mt 自研框架 — 全连接神经网络")
print("=" * 60)
print("  架构: Input(4) -> Linear(4,16) -> ReLU -> Linear(16,8) -> ReLU -> Linear(8,3) -> Softmax")
print("  说明: 两层隐藏层 + 输出层, softmax 激活, CrossEntropy 损失, Adam 优化器")
print("  思考: Iris 数据仅 150 条, 隐藏层神经元不宜过多以减少过拟合。")
print("        Adam 相比 SGD 能自适应学习率, 在梯度各向异性时收敛更快。")

# one-hot 编码标签
y_train_onehot = numpy_one_hot(y_train)
y_test_onehot = numpy_one_hot(y_test)

X_in = Placeholder()
Y_in = Placeholder()

# 构建网络
h1 = mt.core.net.Linear(4, 16, activate_func="relu", init="he_normal")(X_in)
h2 = mt.core.net.Linear(16, 8, activate_func="relu", init="he_normal")(h1)
out = mt.core.net.Linear(8, 3, activate_func="softmax", init="randn")(h2)

loss = measure.CrossEntropy(reduction="mean")(predict=out, label=Y_in)

session = Session()
optimizer = mt.core.optm.Adam(learning_rate=0.05)

EPOCHS = 100
train_losses, test_losses, train_accs, test_accs = [], [], [], []

for epoch in range(EPOCHS):
    # 前向传播 — 训练
    session.run(root_op=loss, feed_dict={X_in: X_train_scaled, Y_in: y_train_onehot})
    train_loss = float(loss.numpy)
    train_losses.append(train_loss)

    y_train_pred = np.argmax(out.numpy, axis=1)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_accs.append(train_acc)

    # 前向传播 — 测试
    session.run(root_op=loss, feed_dict={X_in: X_test_scaled, Y_in: y_test_onehot})
    test_loss = float(loss.numpy)
    test_losses.append(test_loss)

    y_test_pred = np.argmax(out.numpy, axis=1)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_accs.append(test_acc)

    # 反向传播 + 参数更新
    optimizer.backward(loss)
    optimizer.zero_grad()

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"  Epoch {epoch + 1:>3d} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"test_acc: {test_acc:.4f}")

# 最终评估
y_final_pred = np.argmax(out.numpy, axis=1)
final_acc = accuracy_score(y_test, y_final_pred)
print(f"\n  [mt 神经网络] 测试准确率: {final_acc:.4f}")
print(f"  {classification_report(y_test, y_final_pred, target_names=class_names, zero_division=0)}")

print("\n完成！")
