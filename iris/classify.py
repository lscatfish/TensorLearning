"""
鸢尾花分类 — 多种机器学习方法对比

数据集：Iris（150 条，4 特征，3 类）
目标：使用 sklearn 经典算法 + 自研 mt 框架神经网络完成分类，
      并对各算法表现进行分析与思考。

============================================================
算法选择与思考
============================================================

1. **逻辑回归 (Logistic Regression)**
   - 线性分类器，可解释性强。Iris 数据线性可分度高，
     理论上单个 OvR 逻辑回归就能取得不错效果。
   - 局限：对非线性边界无能为力，不适合高维稀疏特征。

2. **支持向量机 (SVM)**
   - 通过核技巧可拟合非线性边界。Iris 数据量小、特征少，
     RBF 核的 SVM 往往是最优选择之一。
   - 局限：对大数据集训练开销大，需要调 C 和 gamma。

3. **随机森林 (Random Forest)**
   - 集成学习，通过多棵决策树投票降低过拟合。
   - 局限：树多了模型大、可解释性下降；小数据集可能不如 SVM。

4. **K 近邻 (KNN)**
   - 无参数、简单直观。Iris 数据集中同类样本特征空间较聚集，
     KNN 天然适合。
   - 局限：预测时要遍历全部训练数据，大规模时慢；对噪声敏感。

5. **自研 mt 神经网络**
   - 两隐藏层全连接网络，使用 Adam 优化器 + CrossEntropy 损失。
     从零实现的自动微分框架，体现对梯度反向传播原理的理解。
   - 局限：150 条数据训练 NN 容易过拟合，需要控制模型复杂度。

============================================================
预期结果排序（从数据特性推断）
============================================================
KNN ≈ SVM > 随机森林 > 逻辑回归 > mt 神经网络
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

import matplotlib
import numpy as np

warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # 无 GUI 后端

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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
idx_to_class = {i: c for i, c in enumerate(class_names)}
y_labels = np.array([class_to_idx[c] for c in y_raw])

print(f"  样本数: {X_raw.shape[0]}")
print(f"  特征数: {X_raw.shape[1]} (花萼长、花萼宽、花瓣长、花瓣宽)")
print(f"  类别数: {len(class_names)} — {list(class_names)}")
print(f"  各类别数量: {[np.sum(y_labels == i) for i in range(3)]}")

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_labels, test_size=0.3, random_state=42, stratify=y_labels
)

# 标准化 — 对线性模型和 NN 很重要
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")

# ============================================================
# 2. sklearn 经典方法
# ============================================================
print("\n" + "=" * 60)
print("2. sklearn 经典机器学习方法")
print("=" * 60)

results = {}

models = {
    "逻辑回归": LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=1000
    ),
    "SVM (RBF核)": SVC(kernel="rbf", C=1.0, gamma="scale"),
    "随机森林": RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    ),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n  [{name}] 测试准确率: {acc:.4f}")
    print(f"  {classification_report(y_test, y_pred, target_names=class_names, zero_division=0)}")

# ============================================================
# 3. mt 自研框架神经网络
# ============================================================
print("\n" + "=" * 60)
print("3. mt 自研框架 — 全连接神经网络")
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
results["mt 神经网络"] = final_acc
print(f"\n  [mt 神经网络] 测试准确率: {final_acc:.4f}")
print(f"  {classification_report(y_test, y_final_pred, target_names=class_names, zero_division=0)}")

# ============================================================
# 4. 结果汇总与分析
# ============================================================
print("\n" + "=" * 60)
print("4. 综合对比与分析")
print("=" * 60)

print("\n  各方法测试准确率排名:")
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for i, (name, acc) in enumerate(sorted_results, 1):
    bar = "█" * int(acc * 50)
    print(f"  {i}. {name:　<12s}  {acc:.4f}  {bar}")

print(f"""\n  +----------------------------------------------------------+\n  |  算法思考总结                                           |\n  +----------------------------------------------------------+\n  | Iris 数据集特点：                                       |\n  |   - 3 类中 setosa 与其他两类完全线性可分                |\n  |   - versicolor 与 virginica 有轻微重叠                  |\n  |   - 仅 150 条数据，4 个连续特征                         |\n  |                                                        |\n  | 为什么 SVM/KNN 通常最优？                              |\n  |   - 小样本 + 低维度是 SVM 的甜点区                      |\n  |   - KNN 无需假设数据分布，特征空间中类内紧密即好用      |\n  |                                                        |\n  | 为什么 NN 在小数据集上不占优？                          |\n  |   - NN 参数多，150 条数据不足以学到泛化模式             |\n  |   - 但通过 L2 正则 / Dropout / 减小模型可改善          |\n  |                                                        |\n  | 为什么逻辑回归也能达到 ~95%？                           |\n  |   - Iris 本质接近线性可分（尤其是 setosa 类）          |\n  |   - 逻辑回归作为 softmax 回归正是最优线性分类器之一    |\n  +----------------------------------------------------------+""")

# ============================================================
# 5. 可视化
# ============================================================
print("5. 生成可视化图表...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Iris 鸢尾花分类 — 多方法对比", fontsize=16, fontweight="bold")

# --- 5a. 损失曲线 ---
ax = axes[0, 0]
ax.plot(train_losses, label="Train Loss", alpha=0.8)
ax.plot(test_losses, label="Test Loss", alpha=0.8)
ax.set_title("mt NN: Loss 曲线")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# --- 5b. 准确率曲线 ---
ax = axes[0, 1]
ax.plot(train_accs, label="Train Acc", alpha=0.8)
ax.plot(test_accs, label="Test Acc", alpha=0.8)
ax.set_title("mt NN: 准确率曲线")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

# --- 5c. 各方法准确率柱状图 ---
ax = axes[0, 2]
names = list(results.keys())
accs = [results[n] * 100 for n in names]
colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
bars = ax.barh(names, accs, color=colors, edgecolor="white")
ax.set_title("各方法测试准确率对比")
ax.set_xlabel("Accuracy (%)")
ax.set_xlim(0, 105)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%", va="center", fontsize=10, fontweight="bold")

# --- 5d. mt NN 混淆矩阵 ---
from sklearn.metrics import ConfusionMatrixDisplay
ax = axes[1, 0]
cm = confusion_matrix(y_test, y_final_pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
    ax=ax, cmap="Blues", colorbar=False
)
ax.set_title("mt NN: 混淆矩阵")

# --- 5e. 随机森林混淆矩阵 ---
ax = axes[1, 1]
rf_model = models["随机森林"]
cm_rf = confusion_matrix(y_test, rf_model.predict(X_test_scaled))
ConfusionMatrixDisplay(cm_rf, display_labels=class_names).plot(
    ax=ax, cmap="Greens", colorbar=False
)
ax.set_title("随机森林: 混淆矩阵")

# --- 5f. 原始数据散点矩阵 ---
ax = axes[1, 2]
ax.axis("off")

# 用 petal_length vs petal_width 展示可分性
scatter_ax = fig.add_axes([0.82, 0.08, 0.15, 0.18])
for i, cls in enumerate(class_names):
    mask = y_labels == i
    scatter_ax.scatter(
        X_raw[mask, 2], X_raw[mask, 3],
        label=cls, alpha=0.7, edgecolors="k", linewidth=0.5, s=30
    )
scatter_ax.set_xlabel("花瓣长", fontsize=8)
scatter_ax.set_ylabel("花瓣宽", fontsize=8)
scatter_ax.set_title("数据分布 (2D 投影)", fontsize=9)
scatter_ax.legend(fontsize=6, framealpha=0.8)
scatter_ax.tick_params(labelsize=6)

plt.tight_layout(rect=[0, 0, 0.8, 0.95])
fig.savefig("iris/results.png", dpi=150, bbox_inches="tight")
print("  图表已保存至: iris/results.png")

plt.close()
print("\n完成！")
