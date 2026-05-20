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
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
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

from matplotlib.gridspec import GridSpec
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA


# ---------- 辅助: 决策边界网格 ----------
def plot_decision_boundary(ax, model, X_2d, y, title, h=0.02):
    """在2D投影上绘制决策边界"""
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolors="k",
                         cmap=plt.cm.RdYlBu, s=60, linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("花瓣长 (标准化)")
    ax.set_ylabel("花瓣宽 (标准化)")
    return scatter


# 选2个最可分特征: 花瓣长(idx=2), 花瓣宽(idx=3)
FEAT2 = [2, 3]
X2_train = X_train_scaled[:, FEAT2]
X2_test = X_test_scaled[:, FEAT2]
X2_all = np.vstack([X2_train, X2_test])
y2_all = np.concatenate([y_train, y_test])


# ============================================================
# 图1: 数据探索 (Data Exploration)
# ============================================================
print("  [1/6] 数据探索...")
fig1 = plt.figure(figsize=(16, 12))
fig1.suptitle("Iris 数据集探索", fontsize=18, fontweight="bold", y=0.98)

gs = GridSpec(3, 3, figure=fig1, hspace=0.4, wspace=0.35)

# 1a. 花瓣长 vs 花瓣宽 散点
ax = fig1.add_subplot(gs[0, :])
for i, cls in enumerate(class_names):
    mask = y_labels == i
    ax.scatter(X_raw[mask, 2], X_raw[mask, 3], label=cls,
               alpha=0.8, edgecolors="k", linewidth=0.5, s=50)
ax.set_xlabel("花瓣长 (cm)")
ax.set_ylabel("花瓣宽 (cm)")
ax.set_title("花瓣长 vs 花瓣宽 (3类分布)", fontsize=14, fontweight="bold")
ax.legend(framealpha=0.8, fontsize=10)
ax.grid(True, alpha=0.3)

# 1b. 类别分布饼图 + 花萼长vs宽
ax = fig1.add_subplot(gs[1, 0])
counts = [np.sum(y_labels == i) for i in range(3)]
colors_pie = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
wedges, texts, autotexts = ax.pie(counts, labels=class_names, autopct="%1.1f%%",
                                   colors=colors_pie, explode=(0.02, 0.02, 0.02))
for at in autotexts:
    at.set_fontsize(10); at.set_fontweight("bold")
ax.set_title("类别分布", fontsize=13, fontweight="bold")

# 1c. 花萼长 vs 花萼宽
ax = fig1.add_subplot(gs[1, 1])
for i, cls in enumerate(class_names):
    mask = y_labels == i
    ax.scatter(X_raw[mask, 0], X_raw[mask, 1], label=cls,
               alpha=0.8, edgecolors="k", linewidth=0.5, s=40)
ax.set_xlabel("花萼长 (cm)"); ax.set_ylabel("花萼宽 (cm)")
ax.set_title("花萼长 vs 花萼宽", fontsize=12, fontweight="bold")
ax.legend(fontsize=7, framealpha=0.8); ax.grid(True, alpha=0.3)

# 1d. 花瓣长 vs 花萼长
ax = fig1.add_subplot(gs[1, 2])
for i, cls in enumerate(class_names):
    mask = y_labels == i
    ax.scatter(X_raw[mask, 2], X_raw[mask, 0], label=cls,
               alpha=0.8, edgecolors="k", linewidth=0.5, s=40)
ax.set_xlabel("花瓣长 (cm)"); ax.set_ylabel("花萼长 (cm)")
ax.set_title("花瓣长 vs 花萼长", fontsize=12, fontweight="bold")
ax.legend(fontsize=7, framealpha=0.8); ax.grid(True, alpha=0.3)

# 1e-h. 四个特征的箱线图 + 直方图
feature_names = ["花萼长 (cm)", "花萼宽 (cm)", "花瓣长 (cm)", "花瓣宽 (cm)"]

# 箱线图 (row 2, span all 3)
ax = fig1.add_subplot(gs[2, :])
box_data = [X_raw[y_labels == i, :] for i in range(3)]
positions = []
labels = []
for i, cls in enumerate(class_names):
    for j, fn in enumerate(feature_names):
        positions.append(i * 5 + j)
        labels.append(f"{cls[:10]}\n{fn}")
bp = ax.boxplot([X_raw[y_labels == i, j] for i in range(3) for j in range(4)],
                positions=positions, patch_artist=True, widths=0.7,
                medianprops=dict(color="black", linewidth=1.5))
for patch, i in zip(bp["boxes"], range(3)):
    patch.set_facecolor(colors_pie[i])
    patch.set_alpha(0.7)
ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
ax.set_ylabel("值 (cm)")
ax.set_title("各类别特征分布 (箱线图)", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

fig1.savefig("iris/01_data_exploration.png", dpi=150, bbox_inches="tight")
plt.close(fig1)


# ============================================================
# 图2: 各模型在2D投影上的决策边界
# ============================================================
print("  [2/6] 决策边界对比...")
fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
fig2.suptitle("各模型决策边界对比 (花瓣长 vs 花瓣宽)", fontsize=16, fontweight="bold")
axes = axes.flatten()

# 2a. 训练数据分布
ax = axes[0]
for i, cls in enumerate(class_names):
    mask_train = y_train == i
    ax.scatter(X2_train[mask_train, 0], X2_train[mask_train, 1],
               label=cls, alpha=0.8, edgecolors="k", linewidth=0.5, s=50)
ax.set_title("训练数据真实分布", fontsize=12, fontweight="bold")
ax.set_xlabel("花瓣长"); ax.set_ylabel("花瓣宽")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 2b. 逻辑回归
lr_2d = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
lr_2d.fit(X2_train, y_train)
scatter = plot_decision_boundary(axes[1], lr_2d, X2_all, y2_all,
                                  f"逻辑回归 (Acc={accuracy_score(y_test, lr_2d.predict(X2_test)):.2%})")

# 2c. SVM
svm_2d = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_2d.fit(X2_train, y_train)
# 标注支持向量
sv_mask = svm_2d.support_
axes[2].scatter(X2_train[sv_mask, 0], X2_train[sv_mask, 1],
                facecolors="none", edgecolors="k", s=120, linewidth=1.5, label="支持向量")
plot_decision_boundary(axes[2], svm_2d, X2_all, y2_all,
                        f"SVM RBF (Acc={accuracy_score(y_test, svm_2d.predict(X2_test)):.2%})")
axes[2].legend(fontsize=8)

# 2d. 随机森林
rf_2d = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_2d.fit(X2_train, y_train)
plot_decision_boundary(axes[3], rf_2d, X2_all, y2_all,
                        f"随机森林 (Acc={accuracy_score(y_test, rf_2d.predict(X2_test)):.2%})")

# 2e. KNN k=1
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X2_train, y_train)
plot_decision_boundary(axes[4], knn1, X2_all, y2_all,
                        f"KNN k=1 (Acc={accuracy_score(y_test, knn1.predict(X2_test)):.2%})")

# 2f. KNN k=15
knn15 = KNeighborsClassifier(n_neighbors=15)
knn15.fit(X2_train, y_train)
plot_decision_boundary(axes[5], knn15, X2_all, y2_all,
                        f"KNN k=15 (Acc={accuracy_score(y_test, knn15.predict(X2_test)):.2%})")

fig2.savefig("iris/02_decision_boundaries.png", dpi=150, bbox_inches="tight")
plt.close(fig2)


# ============================================================
# 图3: 逻辑回归 — 深入分析
# ============================================================
print("  [3/6] 逻辑回归分析...")
fig3 = plt.figure(figsize=(16, 10))
fig3.suptitle("逻辑回归 — 详细分析", fontsize=16, fontweight="bold")

# 3a. 所有特征系数热力图
ax = fig3.add_subplot(2, 3, (1, 2))
coef = models["逻辑回归"].coef_
im = ax.imshow(coef, cmap="RdBu_r", aspect="auto", vmin=-coef.max(), vmax=coef.max())
ax.set_xticks(range(4))
ax.set_xticklabels(feature_names, rotation=20, ha="right")
ax.set_yticks(range(3))
ax.set_yticklabels(class_names)
ax.set_title("逻辑回归系数 (Coefficients)", fontsize=13, fontweight="bold")
for i in range(3):
    for j in range(4):
        ax.text(j, i, f"{coef[i][j]:.2f}", ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if abs(coef[i][j]) > 1 else "black")
fig3.colorbar(im, ax=ax, shrink=0.8)

# 3b. 系数绝对值柱状图
ax = fig3.add_subplot(2, 3, 3)
coef_mean = np.abs(coef).mean(axis=0)
colors_bar = plt.cm.RdYlBu(np.linspace(0.2, 0.8, 4))
bars = ax.bar(range(4), coef_mean, color=colors_bar, edgecolor="white", linewidth=1.5)
ax.set_xticks(range(4))
ax.set_xticklabels(feature_names, rotation=20, ha="right")
ax.set_title("平均特征重要性 (|系数|)", fontsize=13, fontweight="bold")
ax.set_ylabel("平均绝对值")
for bar, val in zip(bars, coef_mean):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

# 3c. 混淆矩阵
ax = fig3.add_subplot(2, 3, 4)
cm_lr = confusion_matrix(y_test, models["逻辑回归"].predict(X_test_scaled))
ConfusionMatrixDisplay(cm_lr, display_labels=class_names).plot(
    ax=ax, cmap="RdYlBu", colorbar=False, text_kw={"fontsize": 13}
)
ax.set_title("混淆矩阵 (测试集)", fontsize=13, fontweight="bold")

# 3d. 每类 Recall/Precision/F1
ax = fig3.add_subplot(2, 3, 5)
from sklearn.metrics import precision_recall_fscore_support
lr_pred = models["逻辑回归"].predict(X_test_scaled)
p, r, f1, _ = precision_recall_fscore_support(y_test, lr_pred, zero_division=0)
x = np.arange(3)
w = 0.25
ax.bar(x - w, p, w, label="Precision", color="#FF6B6B", edgecolor="white")
ax.bar(x, r, w, label="Recall", color="#4ECDC4", edgecolor="white")
ax.bar(x + w, f1, w, label="F1-score", color="#45B7D1", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(class_names)
ax.set_ylim(0, 1.1)
ax.set_title("各类别指标", fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

# 3e. 每类预测概率分布
ax = fig3.add_subplot(2, 3, 6)
proba = models["逻辑回归"].predict_proba(X_test_scaled)
for i in range(3):
    mask = y_test == i
    if mask.sum() > 0:
        ax.hist(proba[mask, i], bins=15, alpha=0.5, label=f"真实: {class_names[i]}",
                color=colors_pie[i], edgecolor="white")
ax.set_xlabel("预测概率")
ax.set_ylabel("样本数")
ax.set_title("各类别预测概率分布", fontsize=13, fontweight="bold")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

fig3.savefig("iris/03_logistic_regression.png", dpi=150, bbox_inches="tight")
plt.close(fig3)


# ============================================================
# 图4: SVM — 深入分析
# ============================================================
print("  [4/6] SVM分析...")
fig4 = plt.figure(figsize=(16, 10))
fig4.suptitle("SVM (RBF核) — 详细分析", fontsize=16, fontweight="bold")

# 4a. SVM 在2D上的决策边界+支持向量
ax = fig4.add_subplot(2, 3, (1, 2))
scatter = plot_decision_boundary(ax, svm_2d, X2_all, y2_all,
                                  f"SVM RBF决策边界 (Acc={accuracy_score(y_test, svm_2d.predict(X2_test)):.2%})")
ax.scatter(X2_train[sv_mask, 0], X2_train[sv_mask, 1],
           facecolors="none", edgecolors="k", s=150, linewidth=2, label=f"支持向量 ({len(sv_mask)}个)")
ax.legend(fontsize=9, loc="upper left")

# 4b. 不同C值的准确率对比
ax = fig4.add_subplot(2, 3, 3)
c_values = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
c_accs = []
for c in c_values:
    sm = SVC(kernel="rbf", C=c, gamma="scale")
    sm.fit(X_train_scaled, y_train)
    c_accs.append(accuracy_score(y_test, sm.predict(X_test_scaled)))
ax.plot(c_values, c_accs, "o-", color="#FF6B6B", linewidth=2, markersize=8, markeredgecolor="white")
ax.set_xscale("log")
ax.set_xlabel("C (正则化参数)")
ax.set_ylabel("测试准确率")
ax.set_title("参数 C 对准确率的影响", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
for c, a in zip(c_values, c_accs):
    ax.annotate(f"C={c}", (c, a), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)

# 4c. 混淆矩阵
ax = fig4.add_subplot(2, 3, 4)
cm_svm = confusion_matrix(y_test, models["SVM (RBF核)"].predict(X_test_scaled))
ConfusionMatrixDisplay(cm_svm, display_labels=class_names).plot(
    ax=ax, cmap="RdPu", colorbar=False, text_kw={"fontsize": 13}
)
ax.set_title("混淆矩阵 (测试集)", fontsize=13, fontweight="bold")

# 4d. 各类别指标
ax = fig4.add_subplot(2, 3, 5)
svm_pred = models["SVM (RBF核)"].predict(X_test_scaled)
p, r, f1, _ = precision_recall_fscore_support(y_test, svm_pred, zero_division=0)
x = np.arange(3); w = 0.25
ax.bar(x - w, p, w, label="Precision", color="#FF6B6B", edgecolor="white")
ax.bar(x, r, w, label="Recall", color="#4ECDC4", edgecolor="white")
ax.bar(x + w, f1, w, label="F1-score", color="#45B7D1", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(class_names)
ax.set_ylim(0, 1.1)
ax.set_title("各类别指标", fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

# 4e. 不同gamma值的对比
ax = fig4.add_subplot(2, 3, 6)
gamma_values = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, "scale", "auto"]
gamma_accs = []
g_labels = []
for g in gamma_values:
    try:
        sm = SVC(kernel="rbf", C=1.0, gamma=g)
        sm.fit(X_train_scaled, y_train)
        gamma_accs.append(accuracy_score(y_test, sm.predict(X_test_scaled)))
        g_labels.append(str(g))
    except Exception:
        pass
colors_g = plt.cm.viridis(np.linspace(0, 1, len(gamma_accs)))
bars = ax.bar(range(len(gamma_accs)), gamma_accs, color=colors_g, edgecolor="white")
ax.set_xticks(range(len(gamma_accs)))
ax.set_xticklabels(g_labels)
ax.set_xlabel("gamma")
ax.set_ylabel("测试准确率")
ax.set_title("参数 gamma 对准确率的影响 (C=1.0)", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis="y")
for bar, a in zip(bars, gamma_accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{a:.3f}", ha="center", fontsize=8, fontweight="bold")

fig4.savefig("iris/04_svm.png", dpi=150, bbox_inches="tight")
plt.close(fig4)


# ============================================================
# 图5: 随机森林 — 深入分析
# ============================================================
print("  [5/6] 随机森林分析...")
fig5 = plt.figure(figsize=(16, 10))
fig5.suptitle("随机森林 — 详细分析", fontsize=16, fontweight="bold")

rf = models["随机森林"]

# 5a. 特征重要性
ax = fig5.add_subplot(2, 3, 1)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
colors_imp = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 4))
bars = ax.bar(range(4), importances[indices], color=colors_imp, edgecolor="white", linewidth=1.5)
ax.set_xticks(range(4))
ax.set_xticklabels([feature_names[i] for i in indices], rotation=20, ha="right")
ax.set_title("特征重要性", fontsize=13, fontweight="bold")
ax.set_ylabel("重要性分数")
for bar, val in zip(bars, importances[indices]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

# 5b. 树数量 vs 准确率
ax = fig5.add_subplot(2, 3, 2)
n_trees = [1, 5, 10, 20, 50, 100, 200]
tree_accs = []
for n in n_trees:
    rf_n = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=42)
    rf_n.fit(X_train_scaled, y_train)
    tree_accs.append(accuracy_score(y_test, rf_n.predict(X_test_scaled)))
ax.plot(n_trees, tree_accs, "o-", color="#45B7D1", linewidth=2, markersize=8, markeredgecolor="white")
ax.set_xlabel("树的数量")
ax.set_ylabel("测试准确率")
ax.set_title("树数量对准确率的影响", fontsize=13, fontweight="bold")
ax.set_xscale("log"); ax.grid(True, alpha=0.3)
for n, a in zip(n_trees, tree_accs):
    ax.annotate(f"{a:.3f}", (n, a), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=9)

# 5c. max_depth vs 准确率
ax = fig5.add_subplot(2, 3, 3)
depths = [1, 2, 3, 5, 7, 10, None]
depth_accs = []
depth_labels = [str(d) for d in depths]
for d in depths:
    rf_d = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42)
    rf_d.fit(X_train_scaled, y_train)
    depth_accs.append(accuracy_score(y_test, rf_d.predict(X_test_scaled)))
ax.plot(range(len(depths)), depth_accs, "o-", color="#FF6B6B", linewidth=2, markersize=8, markeredgecolor="white")
ax.set_xticks(range(len(depths))); ax.set_xticklabels(depth_labels)
ax.set_xlabel("max_depth")
ax.set_ylabel("测试准确率")
ax.set_title("树深度对准确率的影响", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)

# 5d. 混淆矩阵
ax = fig5.add_subplot(2, 3, 4)
cm_rf = confusion_matrix(y_test, rf.predict(X_test_scaled))
ConfusionMatrixDisplay(cm_rf, display_labels=class_names).plot(
    ax=ax, cmap="YlGn", colorbar=False, text_kw={"fontsize": 13}
)
ax.set_title("混淆矩阵 (测试集)", fontsize=13, fontweight="bold")

# 5e. 各类别指标
ax = fig5.add_subplot(2, 3, 5)
rf_pred = rf.predict(X_test_scaled)
p, r, f1, _ = precision_recall_fscore_support(y_test, rf_pred, zero_division=0)
x = np.arange(3); w = 0.25
ax.bar(x - w, p, w, label="Precision", color="#FF6B6B", edgecolor="white")
ax.bar(x, r, w, label="Recall", color="#4ECDC4", edgecolor="white")
ax.bar(x + w, f1, w, label="F1-score", color="#45B7D1", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(class_names)
ax.set_ylim(0, 1.1)
ax.set_title("各类别指标", fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

# 5f. OOB Score (使用oob_score)
ax = fig5.add_subplot(2, 3, 6)
rf_oob = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, oob_score=True)
rf_oob.fit(X_train_scaled, y_train)
ax.axis("off")
info_text = (
    f"随机森林信息:\n\n"
    f"树的数量: 100\n"
    f"最大深度: 5\n"
    f"OOB Score: {rf_oob.oob_score_:.4f}\n"
    f"测试准确率: {accuracy_score(y_test, rf_oob.predict(X_test_scaled)):.4f}\n\n"
    f"特征重要性:\n"
)
for name, imp in zip(feature_names, importances):
    info_text += f"  {name}: {imp:.4f}\n"
ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=11,
        fontfamily="monospace", verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))

fig5.savefig("iris/05_random_forest.png", dpi=150, bbox_inches="tight")
plt.close(fig5)


# ============================================================
# 图6: KNN — 深入分析
# ============================================================
print("  [6/6] KNN + mt NN分析...")
fig6 = plt.figure(figsize=(16, 10))
fig6.suptitle("KNN 与 mt 神经网络 — 详细分析", fontsize=16, fontweight="bold")

# 6a. KNN: k值 vs 准确率
ax = fig6.add_subplot(2, 2, 1)
k_values = range(1, 31)
k_accs = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    k_accs.append(accuracy_score(y_test, knn.predict(X_test_scaled)))
ax.plot(list(k_values), k_accs, "o-", color="#45B7D1", linewidth=2, markersize=5, markeredgecolor="white")
best_k = k_values[np.argmax(k_accs)]
ax.axvline(x=best_k, color="#FF6B6B", linestyle="--", alpha=0.7, label=f"最佳k={best_k}")
ax.set_xlabel("k (邻居数)")
ax.set_ylabel("测试准确率")
ax.set_title("KNN: k值对准确率的影响", fontsize=13, fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)

# 6b. KNN 混淆矩阵
ax = fig6.add_subplot(2, 2, 2)
cm_knn = confusion_matrix(y_test, models["KNN (k=5)"].predict(X_test_scaled))
ConfusionMatrixDisplay(cm_knn, display_labels=class_names).plot(
    ax=ax, cmap="BuPu", colorbar=False, text_kw={"fontsize": 13}
)
ax.set_title("KNN (k=5) 混淆矩阵", fontsize=13, fontweight="bold")

# 6c. mt NN 训练曲线
ax = fig6.add_subplot(2, 2, 3)
ax_loss = ax.twinx()
line1, = ax.plot(train_losses, color="#FF6B6B", linewidth=1.5, label="训练Loss")
line2, = ax.plot(test_losses, color="#FF6B6B", linewidth=1.5, linestyle="--", label="测试Loss")
line3, = ax_loss.plot(train_accs, color="#45B7D1", linewidth=1.5, label="训练Acc")
line4, = ax_loss.plot(test_accs, color="#45B7D1", linewidth=1.5, linestyle="--", label="测试Acc")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss", color="#FF6B6B")
ax_loss.set_ylabel("Accuracy", color="#45B7D1")
ax.set_title("mt NN: 训练曲线", fontsize=13, fontweight="bold")
lines = [line1, line2, line3, line4]
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, fontsize=8, loc="center right")
ax.grid(True, alpha=0.3)

# 6d. mt NN 混淆矩阵
ax = fig6.add_subplot(2, 2, 4)
cm_nn = confusion_matrix(y_test, y_final_pred)
ConfusionMatrixDisplay(cm_nn, display_labels=class_names).plot(
    ax=ax, cmap="Blues", colorbar=False, text_kw={"fontsize": 13}
)
ax.set_title(f"mt NN 混淆矩阵 (Acc={final_acc:.2%})", fontsize=13, fontweight="bold")

fig6.savefig("iris/06_knn_mt_nn.png", dpi=150, bbox_inches="tight")
plt.close(fig6)


# ============================================================
# 图7: 综合对比总结
# ============================================================
print("  [7/7] 综合总结图...")
fig7 = plt.figure(figsize=(20, 14))
fig7.suptitle("Iris 鸢尾花分类 — 多方法综合对比", fontsize=18, fontweight="bold")

gs7 = GridSpec(3, 4, figure=fig7, hspace=0.5, wspace=0.4)

# 7a. 准确率柱状图
ax = fig7.add_subplot(gs7[0, :2])
method_names = list(results.keys())
method_accs = [results[n] * 100 for n in method_names]
colors_top = plt.cm.Set3(np.linspace(0, 1, len(method_names)))
bars = ax.barh(method_names, method_accs, color=colors_top, edgecolor="white", height=0.6)
ax.set_title("各方法测试准确率对比", fontsize=15, fontweight="bold")
ax.set_xlabel("准确率 (%)")
ax.set_xlim(0, 108)
for bar, acc in zip(bars, method_accs):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%", va="center", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")

# 7b. 每类F1-Score对比
ax = fig7.add_subplot(gs7[0, 2:])
f1_scores = {}
for name, model in {**models, "mt 神经网络": None}.items():
    if name == "mt 神经网络":
        pred = y_final_pred
    else:
        pred = model.predict(X_test_scaled)
    _, _, f1, _ = precision_recall_fscore_support(y_test, pred, zero_division=0)
    f1_scores[name] = f1
x = np.arange(3); w = 0.15
for idx, (name, f1s) in enumerate(f1_scores.items()):
    ax.bar(x + idx * w - w * 2, f1s, w, label=name, alpha=0.85, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(class_names)
ax.set_ylim(0, 1.15)
ax.set_title("各类别 F1-Score 对比", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3, axis="y")

# 7c. 所有方法混淆矩阵一览
cm_methods = [
    ("逻辑回归", confusion_matrix(y_test, models["逻辑回归"].predict(X_test_scaled))),
    ("SVM RBF", confusion_matrix(y_test, models["SVM (RBF核)"].predict(X_test_scaled))),
    ("随机森林", confusion_matrix(y_test, models["随机森林"].predict(X_test_scaled))),
    ("KNN k=5", confusion_matrix(y_test, models["KNN (k=5)"].predict(X_test_scaled))),
    ("mt NN", confusion_matrix(y_test, y_final_pred)),
]
cm_positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
for (name, cm), (row, col) in zip(cm_methods, cm_positions):
    ax = fig7.add_subplot(gs7[row, col])
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="YlOrRd", colorbar=False, text_kw={"fontsize": 10},
        im_kw={"vmin": 0, "vmax": 15}
    )
    ax.set_title(name, fontsize=11, fontweight="bold")

fig7.tight_layout()
fig7.savefig("iris/07_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig7)

print("  图表已保存至 iris/ 目录:")
print("    01_data_exploration.png  — 数据探索")
print("    02_decision_boundaries.png — 决策边界对比")
print("    03_logistic_regression.png — 逻辑回归分析")
print("    04_svm.png — SVM分析")
print("    05_random_forest.png — 随机森林分析")
print("    06_knn_mt_nn.png — KNN + mt NN分析")
print("    07_summary.png — 综合对比总结")

print("\n完成！")
