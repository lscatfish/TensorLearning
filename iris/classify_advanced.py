"""
鸢尾花分类 — 深入机器学习方法对比（mt 无关版本）

数据集：Iris（150 条，4 特征，3 类）
目标：使用 sklearn 经典算法 + 高级模型完成分类，
       重点关注 MLP 及其他深入架构的对比分析。

============================================================
算法选择与思考
============================================================

1. **逻辑回归 (Logistic Regression)**
   - 线性分类器，可解释性强。Iris 数据线性可分度高，
     理论上单个 OvR 逻辑回归就能取得不错效果。

2. **支持向量机 (SVM)**
   - 通过核技巧可拟合非线性边界。Iris 数据量小、特征少，
     RBF 核的 SVM 往往是最优选择之一。

3. **随机森林 (Random Forest)**
   - 集成学习，通过多棵决策树投票降低过拟合。

4. **K 近邻 (KNN)**
   - 无参数、简单直观。Iris 数据集中同类样本特征空间较聚集，
     KNN 天然适合。

5. **MLP 多层感知器 (sklearn)**
   - 使用 sklearn.neural_network.MLPClassifier
   - 支持多层全连接网络 + Adam 优化 + ReLU 激活
   - 相比 mt 自研框架，MLPClassifier 内置了 L2 正则、早停、
     自适应学习率等实用特性，能更好应对小数据集过拟合问题。
   - 思考：调节 hidden_layer_sizes / alpha / early_stopping
     可显著影响小数据集上的泛化性能。

6. **梯度提升树 (Gradient Boosting)**
   - 基于残差逐步迭代的集成方法，通常在结构化数据上表现优异。
   - 相比随机森林，GBDT 能更细致地拟合数据，
     但对超参数（lr, n_estimators, max_depth）更敏感。

7. **决策树 (Decision Tree)**
   - 单一决策树，可解释性最强，但容易过拟合。
     作为对比基线，展示剪枝（max_depth）的影响。

8. **高斯朴素贝叶斯 (Gaussian Naive Bayes)**
   - 基于贝叶斯定理 + 特征独立假设。
     虽然特征独立假设在 Iris 上不完全成立，
     但花瓣特征区分度高，仍能取得不错效果。

============================================================
预期结果排序（从数据特性推断）
============================================================
SVM ≈ KNN > MLP (调优后) ≈ GradientBoosting > 随机森林 > 逻辑回归 > 决策树 > 朴素贝叶斯
"""

import warnings
import numpy as np
import matplotlib

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"  # SVG 中文字保留为文本
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import os

os.makedirs("iris/output", exist_ok=True)

# ============================================================
# 1. 数据加载与预处理
# ============================================================
print("=" * 60)
print("1. 加载 Iris 数据集")
print("=" * 60)

DATA_PATH = os.path.join(os.path.dirname(__file__), "iris.data")
data = np.loadtxt(DATA_PATH, delimiter=",", dtype=str)

X_raw = data[:, :-1].astype(np.float64)
y_raw = data[:, -1]

class_names = np.unique(y_raw)
class_to_idx = {c: i for i, c in enumerate(class_names)}
idx_to_class = {i: c for i, c in enumerate(class_names)}
y_labels = np.array([class_to_idx[c] for c in y_raw])

print(f"  样本数: {X_raw.shape[0]}")
print(f"  特征数: {X_raw.shape[1]} (花萼长、花萼宽、花瓣长、花瓣宽)")
print(f"  类别数: {len(class_names)} — {list(class_names)}")
print(f"  各类别数量: {[np.sum(y_labels == i) for i in range(3)]}")

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_labels, test_size=0.3, random_state=42, stratify=y_labels
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")

# ============================================================
# 2. 模型定义与训练
# ============================================================
print("\n" + "=" * 60)
print("2. 机器学习模型训练与评估")
print("=" * 60)

results = {}
models = {
    "逻辑回归": LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42
    ),
    "SVM (RBF核)": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
    "随机森林": RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    ),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    # MLP — 小型网络，加正则化防过拟合
    "MLP (16,8)": MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        alpha=0.001,
        max_iter=1000,
        random_state=42,
    ),
    # MLP — 更浅层，尝试单隐藏层的效果
    "MLP (32,)": MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="adam",
        alpha=0.001,
        max_iter=1000,
        random_state=42,
    ),
    # MLP — 深层窄网络，观察过拟合现象
    "MLP (64,32,16)": MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        alpha=0.01,
        max_iter=1000,
        random_state=42,
    ),
    "梯度提升树": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    ),
    "决策树": DecisionTreeClassifier(max_depth=3, random_state=42),
    "高斯朴素贝叶斯": GaussianNB(),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n  [{name}] 测试准确率: {acc:.4f}")
    print(f"  {classification_report(y_test, y_pred, target_names=class_names, zero_division=0)}")

# ============================================================
# 3. MLP 深入调优 — GridSearch
# ============================================================
print("\n" + "=" * 60)
print("3. MLP 超参数调优 (GridSearchCV)")
print("=" * 60)

param_grid = {
    "hidden_layer_sizes": [(16,), (32,), (16, 8), (32, 16), (64, 32, 16)],
    "alpha": [0.0001, 0.001, 0.01, 0.1],
    "activation": ["relu", "tanh"],
}

grid_search = GridSearchCV(
    MLPClassifier(solver="adam", max_iter=1000, random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train_scaled, y_train)

best_mlp = grid_search.best_estimator_
y_pred_best_mlp = best_mlp.predict(X_test_scaled)
best_mlp_acc = accuracy_score(y_test, y_pred_best_mlp)
results["MLP (GridSearch最佳)"] = best_mlp_acc

print(f"\n  最佳参数: {grid_search.best_params_}")
print(f"  最佳交叉验证分数: {grid_search.best_score_:.4f}")
print(f"  测试准确率: {best_mlp_acc:.4f}")
print(f"  {classification_report(y_test, y_pred_best_mlp, target_names=class_names, zero_division=0)}")

# ============================================================
# 4. 交叉验证对比
# ============================================================
print("\n" + "=" * 60)
print("4. 5折交叉验证对比")
print("=" * 60)

cv_models = {
    "逻辑回归": LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42),
    "SVM (RBF核)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
    "随机森林": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "MLP (GridSearch最佳)": best_mlp,
    "梯度提升树": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "高斯朴素贝叶斯": GaussianNB(),
}
cv_results = {}
for name, model in cv_models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    cv_results[name] = (scores.mean(), scores.std())
    print(f"  [{name}] CV准确率: {scores.mean():.4f} ± {scores.std():.4f}")

# ============================================================
# 5. 结果汇总与分析
# ============================================================
print("\n" + "=" * 60)
print("5. 综合对比与分析")
print("=" * 60)

print("\n  各方法测试准确率排名:")
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for i, (name, acc) in enumerate(sorted_results, 1):
    bar = "█" * int(acc * 50)
    print(f"  {i}. {name:　<16s}  {acc:.4f}  {bar}")

print(f"""\n  +--------------------------------------------------------------+
  |  算法思考总结                                             |
  +--------------------------------------------------------------+
  | MLP vs 传统方法:                                          |
  |   - MLP 在调优后能达到与传统方法相当的准确率                 |
  |   - 小数据下 MLP 容易过拟合, 需配合早停 + L2 正则          |
  |   - 隐层大小 (16,8) 比 (32,) 略好 — 深度比宽度更优?        |
  |   - GridSearch 帮助找到最适合 Iris 的架构                  |
  |                                                            |
  | 为什么集成方法表现稳定?                                    |
  |   - 随机森林和梯度提升树通过集成降低方差/偏差              |
  |   - Iris 数据小, 单模型可能不稳定, 集成后更鲁棒            |
  |                                                            |
  | 朴素贝叶斯假设的局限性:                                    |
  |   - 花瓣长 vs 花瓣宽强相关, 违反独立假设                   |
  |   - 但特征区分度高, 仍然有竞争力                           |
  +--------------------------------------------------------------+""")

# ============================================================
# 6. 可视化
# ============================================================
print("\n6. 生成可视化图表...")

feature_names = ["花萼长 (cm)", "花萼宽 (cm)", "花瓣长 (cm)", "花瓣宽 (cm)"]
colors_pie = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

# ---------- 辅助: 决策边界网格 ----------
def plot_decision_boundary(ax, model, X_2d, y, title, h=0.02):
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
# 图1: 数据探索
# ============================================================
print("  [1/8] 数据探索...")
fig1 = plt.figure(figsize=(16, 12))
fig1.suptitle("Iris 数据集探索", fontsize=18, fontweight="bold", y=0.98)

gs = GridSpec(3, 3, figure=fig1, hspace=0.4, wspace=0.35)

ax = fig1.add_subplot(gs[0, :])
for i, cls in enumerate(class_names):
    mask = y_labels == i
    ax.scatter(X_raw[mask, 2], X_raw[mask, 3], label=cls,
               alpha=0.8, edgecolors="k", linewidth=0.5, s=50)
ax.set_xlabel("花瓣长 (cm)"); ax.set_ylabel("花瓣宽 (cm)")
ax.set_title("花瓣长 vs 花瓣宽 (3类分布)", fontsize=14, fontweight="bold")
ax.legend(framealpha=0.8, fontsize=10); ax.grid(True, alpha=0.3)

ax = fig1.add_subplot(gs[1, 0])
counts = [np.sum(y_labels == i) for i in range(3)]
wedges, texts, autotexts = ax.pie(counts, labels=class_names, autopct="%1.1f%%",
                                   colors=colors_pie, explode=(0.02, 0.02, 0.02))
for at in autotexts:
    at.set_fontsize(10); at.set_fontweight("bold")
ax.set_title("类别分布", fontsize=13, fontweight="bold")

ax = fig1.add_subplot(gs[1, 1])
for i, cls in enumerate(class_names):
    mask = y_labels == i
    ax.scatter(X_raw[mask, 0], X_raw[mask, 1], label=cls,
               alpha=0.8, edgecolors="k", linewidth=0.5, s=40)
ax.set_xlabel("花萼长 (cm)"); ax.set_ylabel("花萼宽 (cm)")
ax.set_title("花萼长 vs 花萼宽", fontsize=12, fontweight="bold")
ax.legend(fontsize=7, framealpha=0.8); ax.grid(True, alpha=0.3)

ax = fig1.add_subplot(gs[1, 2])
for i, cls in enumerate(class_names):
    mask = y_labels == i
    ax.scatter(X_raw[mask, 2], X_raw[mask, 0], label=cls,
               alpha=0.8, edgecolors="k", linewidth=0.5, s=40)
ax.set_xlabel("花瓣长 (cm)"); ax.set_ylabel("花萼长 (cm)")
ax.set_title("花瓣长 vs 花萼长", fontsize=12, fontweight="bold")
ax.legend(fontsize=7, framealpha=0.8); ax.grid(True, alpha=0.3)

ax = fig1.add_subplot(gs[2, :])
box_data = [X_raw[y_labels == i, :] for i in range(3)]
positions = []; labels = []
for i, cls in enumerate(class_names):
    for j, fn in enumerate(feature_names):
        positions.append(i * 5 + j)
        labels.append(f"{cls[:10]}\n{fn}")
bp = ax.boxplot([X_raw[y_labels == i, j] for i in range(3) for j in range(4)],
                positions=positions, patch_artist=True, widths=0.7,
                medianprops=dict(color="black", linewidth=1.5))
for patch, i in zip(bp["boxes"], range(3)):
    patch.set_facecolor(colors_pie[i]); patch.set_alpha(0.7)
ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
ax.set_ylabel("值 (cm)")
ax.set_title("各类别特征分布 (箱线图)", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

fig1.savefig("iris/output/01_data_exploration_advanced.svg", bbox_inches="tight")
plt.close(fig1)

# ============================================================
# 图2: 决策边界对比 — 8个子图
# ============================================================
print("  [2/8] 决策边界对比 (8模型)...")
fig2, axes = plt.subplots(2, 4, figsize=(24, 12))
fig2.suptitle("各模型决策边界对比 (花瓣长 vs 花瓣宽)", fontsize=16, fontweight="bold")
axes = axes.flatten()

# 训练真实分布
ax = axes[0]
for i, cls in enumerate(class_names):
    mask_train = y_train == i
    ax.scatter(X2_train[mask_train, 0], X2_train[mask_train, 1],
               label=cls, alpha=0.8, edgecolors="k", linewidth=0.5, s=50)
ax.set_title("训练数据真实分布", fontsize=12, fontweight="bold")
ax.set_xlabel("花瓣长"); ax.set_ylabel("花瓣宽")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 其他7个模型
boundary_models = {
    "逻辑回归": LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42),
    "SVM RBF": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
    "随机森林": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "KNN k=5": KNeighborsClassifier(n_neighbors=5),
    "MLP (16,8)": MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
                                 alpha=0.001, max_iter=1000, random_state=42),
    "梯度提升树": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=42),
    "决策树 (d=3)": DecisionTreeClassifier(max_depth=3, random_state=42),
}

for idx, (name, model) in enumerate(boundary_models.items()):
    ax = axes[idx + 1]
    model.fit(X2_train, y_train)
    acc = accuracy_score(y_test, model.predict(X2_test))
    plot_decision_boundary(ax, model, X2_all, y2_all, f"{name} (Acc={acc:.2%})")

fig2.savefig("iris/output/02_decision_boundaries_advanced.svg", bbox_inches="tight")
plt.close(fig2)

# ============================================================
# 图3: 逻辑回归分析
# ============================================================
print("  [3/8] 逻辑回归分析...")
fig3 = plt.figure(figsize=(16, 10))
fig3.suptitle("逻辑回归 — 详细分析", fontsize=16, fontweight="bold")

ax = fig3.add_subplot(2, 3, (1, 2))
coef = models["逻辑回归"].coef_
im = ax.imshow(coef, cmap="RdBu_r", aspect="auto", vmin=-coef.max(), vmax=coef.max())
ax.set_xticks(range(4)); ax.set_xticklabels(feature_names, rotation=20, ha="right")
ax.set_yticks(range(3)); ax.set_yticklabels(class_names)
ax.set_title("逻辑回归系数", fontsize=13, fontweight="bold")
for i in range(3):
    for j in range(4):
        ax.text(j, i, f"{coef[i][j]:.2f}", ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if abs(coef[i][j]) > 1 else "black")
fig3.colorbar(im, ax=ax, shrink=0.8)

ax = fig3.add_subplot(2, 3, 3)
coef_mean = np.abs(coef).mean(axis=0)
colors_bar = plt.cm.RdYlBu(np.linspace(0.2, 0.8, 4))
bars = ax.bar(range(4), coef_mean, color=colors_bar, edgecolor="white", linewidth=1.5)
ax.set_xticks(range(4)); ax.set_xticklabels(feature_names, rotation=20, ha="right")
ax.set_title("平均特征重要性 (|系数|)", fontsize=13, fontweight="bold")
ax.set_ylabel("平均绝对值")
for bar, val in zip(bars, coef_mean):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

ax = fig3.add_subplot(2, 3, 4)
cm_lr = confusion_matrix(y_test, models["逻辑回归"].predict(X_test_scaled))
ConfusionMatrixDisplay(cm_lr, display_labels=class_names).plot(
    ax=ax, cmap="RdYlBu", colorbar=False, text_kw={"fontsize": 13})
ax.set_title("混淆矩阵 (测试集)", fontsize=13, fontweight="bold")

ax = fig3.add_subplot(2, 3, 5)
lr_pred = models["逻辑回归"].predict(X_test_scaled)
p, r, f1, _ = precision_recall_fscore_support(y_test, lr_pred, zero_division=0)
x = np.arange(3); w = 0.25
ax.bar(x - w, p, w, label="Precision", color="#FF6B6B", edgecolor="white")
ax.bar(x, r, w, label="Recall", color="#4ECDC4", edgecolor="white")
ax.bar(x + w, f1, w, label="F1-score", color="#45B7D1", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(class_names)
ax.set_ylim(0, 1.1)
ax.set_title("各类别指标", fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

ax = fig3.add_subplot(2, 3, 6)
proba = models["逻辑回归"].predict_proba(X_test_scaled)
for i in range(3):
    mask = y_test == i
    if mask.sum() > 0:
        ax.hist(proba[mask, i], bins=15, alpha=0.5, label=f"真实: {class_names[i]}",
                color=colors_pie[i], edgecolor="white")
ax.set_xlabel("预测概率"); ax.set_ylabel("样本数")
ax.set_title("各类别预测概率分布", fontsize=13, fontweight="bold")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

fig3.savefig("iris/output/03_logistic_regression_advanced.svg", bbox_inches="tight")
plt.close(fig3)

# ============================================================
# 图4: MLP 深入分析
# ============================================================
print("  [4/8] MLP 深入分析...")
fig4 = plt.figure(figsize=(18, 12))
fig4.suptitle("MLP (多层感知器) — 深入分析", fontsize=16, fontweight="bold")

# 4a. 不同架构准确率对比
ax = fig4.add_subplot(2, 3, 1)
arch_names = ["MLP (16,8)", "MLP (32,)", "MLP (64,32,16)", "MLP (GridSearch最佳)"]
arch_accs = [results.get(n, 0) for n in arch_names]
colors_arch = plt.cm.viridis(np.linspace(0.2, 0.8, len(arch_names)))
bars = ax.barh(arch_names, [a * 100 for a in arch_accs], color=colors_arch, edgecolor="white")
ax.set_xlabel("测试准确率 (%)")
ax.set_title("不同 MLP 架构对比", fontsize=13, fontweight="bold")
ax.set_xlim(0, 105)
for bar, acc in zip(bars, arch_accs):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{acc:.2%}", va="center", fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")

# 4b. MLP 损失曲线
ax = fig4.add_subplot(2, 3, 2)
mlp_for_plot = MLPClassifier(
    hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
    alpha=0.001, max_iter=1000, random_state=42
)
mlp_for_plot.fit(X_train_scaled, y_train)
ax.plot(mlp_for_plot.loss_curve_, label="训练损失", color="#FF6B6B", linewidth=2)
if hasattr(mlp_for_plot, "validation_scores_") and mlp_for_plot.validation_scores_:
    val_scores = [1 - s for s in mlp_for_plot.validation_scores_]
    ax.plot(val_scores, label="验证损失 (1-acc)", color="#45B7D1", linewidth=2, linestyle="--")
ax.set_xlabel("迭代次数"); ax.set_ylabel("损失")
ax.set_title("MLP (16,8) 训练曲线", fontsize=13, fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)

# 4c. MLP 混淆矩阵 (GridSearch最佳)
ax = fig4.add_subplot(2, 3, 3)
cm_mlp = confusion_matrix(y_test, y_pred_best_mlp)
ConfusionMatrixDisplay(cm_mlp, display_labels=class_names).plot(
    ax=ax, cmap="Blues", colorbar=False, text_kw={"fontsize": 13})
ax.set_title(f"MLP (GridSearch最佳) 混淆矩阵\nacc={best_mlp_acc:.2%}", fontsize=12, fontweight="bold")

# 4d. GridSearch 热力图 (alpha vs hidden_layer_sizes)
ax = fig4.add_subplot(2, 3, 4)
cv_results_grid = grid_search.cv_results_
mean_scores = cv_results_grid["mean_test_score"]
param_combos = len(param_grid["hidden_layer_sizes"]) * len(param_grid["alpha"])
score_matrix = mean_scores[:param_combos].reshape(
    len(param_grid["hidden_layer_sizes"]), len(param_grid["alpha"])
)
im = ax.imshow(score_matrix, cmap="YlOrRd", aspect="auto", vmin=0.8, vmax=1.0)
ax.set_xticks(range(len(param_grid["alpha"])))
ax.set_xticklabels([f"α={a}" for a in param_grid["alpha"]])
ax.set_yticks(range(len(param_grid["hidden_layer_sizes"])))
ax.set_yticklabels(param_grid["hidden_layer_sizes"])
ax.set_title("GridSearch: CV准确率", fontsize=13, fontweight="bold")
for i in range(score_matrix.shape[0]):
    for j in range(score_matrix.shape[1]):
        ax.text(j, i, f"{score_matrix[i, j]:.3f}", ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white" if score_matrix[i, j] > 0.95 else "black")
fig4.colorbar(im, ax=ax, shrink=0.8)

# 4e. 不同 alpha 的影响
ax = fig4.add_subplot(2, 3, 5)
alpha_values = [0.0001, 0.001, 0.01, 0.1, 1.0]
alpha_scores = []
for a in alpha_values:
    mlp_a = MLPClassifier(
        hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
        alpha=a, max_iter=1000, random_state=42
    )
    mlp_a.fit(X_train_scaled, y_train)
    alpha_scores.append(accuracy_score(y_test, mlp_a.predict(X_test_scaled)))
ax.plot([str(a) for a in alpha_values], alpha_scores, "o-", color="#4ECDC4",
        linewidth=2, markersize=10, markeredgecolor="white")
ax.set_xlabel("L2正则强度 (alpha)"); ax.set_ylabel("测试准确率")
ax.set_title("正则强度对 MLP 影响", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
for a, s in zip(alpha_values, alpha_scores):
    ax.annotate(f"{s:.3f}", (str(a), s), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9)

# 4f. activation 函数对比
ax = fig4.add_subplot(2, 3, 6)
activations = ["relu", "tanh", "logistic"]
act_scores = []
for act in activations:
    mlp_act = MLPClassifier(
        hidden_layer_sizes=(16, 8), activation=act, solver="adam",
        alpha=0.001, max_iter=1000, random_state=42
    )
    mlp_act.fit(X_train_scaled, y_train)
    act_scores.append(accuracy_score(y_test, mlp_act.predict(X_test_scaled)))
colors_act = plt.cm.Set2(np.linspace(0, 1, len(activations)))
bars = ax.bar(activations, [s * 100 for s in act_scores], color=colors_act, edgecolor="white")
ax.set_ylabel("测试准确率 (%)"); ax.set_ylim(80, 105)
ax.set_title("激活函数对比", fontsize=13, fontweight="bold")
for bar, s in zip(bars, act_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{s:.2%}", ha="center", fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

fig4.savefig("iris/output/04_mlp_analysis.svg", bbox_inches="tight")
plt.close(fig4)

# ============================================================
# 图5: 随机森林 + 梯度提升树对比
# ============================================================
print("  [5/8] 集成方法分析...")
fig5 = plt.figure(figsize=(18, 12))
fig5.suptitle("集成方法 — 随机森林 vs 梯度提升树", fontsize=16, fontweight="bold")

# 5a. 特征重要性对比
ax = fig5.add_subplot(2, 3, 1)
rf = models["随机森林"]
importances_rf = rf.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1]
bars = ax.bar(range(4), importances_rf[indices_rf],
              color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, 4)),
              edgecolor="white", linewidth=1.5)
ax.set_xticks(range(4))
ax.set_xticklabels([feature_names[i] for i in indices_rf], rotation=20, ha="right")
ax.set_title("随机森林特征重要性", fontsize=13, fontweight="bold")
ax.set_ylabel("重要性分数")
for bar, val in zip(bars, importances_rf[indices_rf]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

# 5b. 梯度提升树特征重要性
ax = fig5.add_subplot(2, 3, 2)
gbdt = models["梯度提升树"]
importances_gbdt = gbdt.feature_importances_
indices_gbdt = np.argsort(importances_gbdt)[::-1]
bars = ax.bar(range(4), importances_gbdt[indices_gbdt],
              color=plt.cm.PuBu(np.linspace(0.3, 0.9, 4)),
              edgecolor="white", linewidth=1.5)
ax.set_xticks(range(4))
ax.set_xticklabels([feature_names[i] for i in indices_gbdt], rotation=20, ha="right")
ax.set_title("梯度提升树特征重要性", fontsize=13, fontweight="bold")
ax.set_ylabel("重要性分数")
for bar, val in zip(bars, importances_gbdt[indices_gbdt]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

# 5c. 树数量影响对比
ax = fig5.add_subplot(2, 3, 3)
n_estimators_range = [1, 5, 10, 20, 50, 100, 200]
rf_accs, gbdt_accs = [], []
for n in n_estimators_range:
    rf_n = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=42)
    rf_n.fit(X_train_scaled, y_train)
    rf_accs.append(accuracy_score(y_test, rf_n.predict(X_test_scaled)))

    gbdt_n = GradientBoostingClassifier(n_estimators=n, learning_rate=0.1, max_depth=2, random_state=42)
    gbdt_n.fit(X_train_scaled, y_train)
    gbdt_accs.append(accuracy_score(y_test, gbdt_n.predict(X_test_scaled)))

ax.plot(n_estimators_range, rf_accs, "o-", color="#FF6B6B", linewidth=2,
        markersize=8, markeredgecolor="white", label="随机森林")
ax.plot(n_estimators_range, gbdt_accs, "s-", color="#45B7D1", linewidth=2,
        markersize=8, markeredgecolor="white", label="梯度提升树")
ax.set_xscale("log"); ax.set_xlabel("估计器数量")
ax.set_ylabel("测试准确率")
ax.set_title("估计器数量影响对比", fontsize=13, fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)

# 5d. 随机森林混淆矩阵
ax = fig5.add_subplot(2, 3, 4)
cm_rf = confusion_matrix(y_test, models["随机森林"].predict(X_test_scaled))
ConfusionMatrixDisplay(cm_rf, display_labels=class_names).plot(
    ax=ax, cmap="YlGn", colorbar=False, text_kw={"fontsize": 13})
ax.set_title("随机森林混淆矩阵", fontsize=13, fontweight="bold")

# 5e. 梯度提升树混淆矩阵
ax = fig5.add_subplot(2, 3, 5)
cm_gbdt = confusion_matrix(y_test, models["梯度提升树"].predict(X_test_scaled))
ConfusionMatrixDisplay(cm_gbdt, display_labels=class_names).plot(
    ax=ax, cmap="PuBu", colorbar=False, text_kw={"fontsize": 13})
ax.set_title("梯度提升树混淆矩阵", fontsize=13, fontweight="bold")

# 5f. 梯度提升树学习率影响
ax = fig5.add_subplot(2, 3, 6)
lr_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
lr_accs = []
for lr in lr_values:
    gbdt_lr = GradientBoostingClassifier(n_estimators=100, learning_rate=lr, max_depth=2, random_state=42)
    gbdt_lr.fit(X_train_scaled, y_train)
    lr_accs.append(accuracy_score(y_test, gbdt_lr.predict(X_test_scaled)))
ax.plot([str(lr) for lr in lr_values], lr_accs, "o-", color="#4ECDC4",
        linewidth=2, markersize=10, markeredgecolor="white")
ax.set_xlabel("学习率"); ax.set_ylabel("测试准确率")
ax.set_title("学习率对 GBDT 影响", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
for lr, s in zip(lr_values, lr_accs):
    ax.annotate(f"{s:.3f}", (str(lr), s), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9)

fig5.savefig("iris/output/05_ensemble_methods.svg", bbox_inches="tight")
plt.close(fig5)

# ============================================================
# 图6: 决策树可视化 + 朴素贝叶斯分析
# ============================================================
print("  [6/8] 决策树与朴素贝叶斯分析...")
fig6 = plt.figure(figsize=(18, 12))
fig6.suptitle("决策树 & 高斯朴素贝叶斯", fontsize=16, fontweight="bold")

# 6a. 决策树可视化 (训练全部4个特征)
ax = fig6.add_subplot(2, 3, (1, 2))
dt_viz = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_viz.fit(X_train_scaled, y_train)
plot_tree(dt_viz, feature_names=feature_names, class_names=list(class_names),
          filled=True, rounded=True, ax=ax, fontsize=9)
ax.set_title("决策树 (max_depth=3)", fontsize=13, fontweight="bold")

# 6b. 不同深度的影响
ax = fig6.add_subplot(2, 3, 3)
depths = range(1, 11)
depth_accs_train, depth_accs_test = [], []
for d in depths:
    dt_d = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt_d.fit(X_train_scaled, y_train)
    depth_accs_train.append(accuracy_score(y_train, dt_d.predict(X_train_scaled)))
    depth_accs_test.append(accuracy_score(y_test, dt_d.predict(X_test_scaled)))
ax.plot(list(depths), depth_accs_train, "o-", color="#FF6B6B", label="训练准确率")
ax.plot(list(depths), depth_accs_test, "s-", color="#45B7D1", label="测试准确率")
ax.axvline(x=3, color="gray", linestyle="--", alpha=0.5, label="max_depth=3")
ax.set_xlabel("max_depth"); ax.set_ylabel("准确率")
ax.set_title("树深度对过拟合的影响", fontsize=13, fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)

# 6c. 朴素贝叶�: 各类别高斯分布参数
ax = fig6.add_subplot(2, 3, 4)
gnb = models["高斯朴素贝叶斯"]
theta = gnb.theta_
sigma = gnb.var_
for i, cls in enumerate(class_names):
    ax.errorbar(range(4), theta[i], yerr=np.sqrt(sigma[i]),
                fmt="o-", label=cls, capsize=5, capthick=2,
                color=colors_pie[i], markersize=8)
ax.set_xticks(range(4)); ax.set_xticklabels(feature_names, rotation=20, ha="right")
ax.set_title("各类别特征均值 ± 标准差", fontsize=12, fontweight="bold")
ax.set_ylabel("标准化值"); ax.legend(); ax.grid(True, alpha=0.3)

# 6d. 朴素贝叶斯混淆矩阵
ax = fig6.add_subplot(2, 3, 5)
cm_gnb = confusion_matrix(y_test, models["高斯朴素贝叶斯"].predict(X_test_scaled))
ConfusionMatrixDisplay(cm_gnb, display_labels=class_names).plot(
    ax=ax, cmap="OrRd", colorbar=False, text_kw={"fontsize": 13})
ax.set_title("高斯朴素贝叶斯混淆矩阵", fontsize=13, fontweight="bold")

# 6e. 决策树混淆矩阵
ax = fig6.add_subplot(2, 3, 6)
cm_dt = confusion_matrix(y_test, models["决策树"].predict(X_test_scaled))
ConfusionMatrixDisplay(cm_dt, display_labels=class_names).plot(
    ax=ax, cmap="YlOrBr", colorbar=False, text_kw={"fontsize": 13})
ax.set_title("决策树 (d=3) 混淆矩阵", fontsize=13, fontweight="bold")

fig6.savefig("iris/output/06_tree_nb_analysis.svg", bbox_inches="tight")
plt.close(fig6)

# ============================================================
# 图7: KNN + MLP 训练曲线对比
# ============================================================
print("  [7/8] KNN + MLP 训练曲线对比...")
fig7 = plt.figure(figsize=(16, 10))
fig7.suptitle("KNN 与 MLP — 详细分析", fontsize=16, fontweight="bold")

# 7a. KNN k值扫描
ax = fig7.add_subplot(2, 2, 1)
k_values = range(1, 31)
k_accs = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    k_accs.append(accuracy_score(y_test, knn.predict(X_test_scaled)))
ax.plot(list(k_values), k_accs, "o-", color="#45B7D1", linewidth=2, markersize=5, markeredgecolor="white")
best_k = k_values[np.argmax(k_accs)]
ax.axvline(x=best_k, color="#FF6B6B", linestyle="--", alpha=0.7, label=f"最佳k={best_k}")
ax.set_xlabel("k (邻居数)"); ax.set_ylabel("测试准确率")
ax.set_title("KNN: k值对准确率的影响", fontsize=13, fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)

# 7b. KNN 混淆矩阵 (k=5)
ax = fig7.add_subplot(2, 2, 2)
cm_knn = confusion_matrix(y_test, models["KNN (k=5)"].predict(X_test_scaled))
ConfusionMatrixDisplay(cm_knn, display_labels=class_names).plot(
    ax=ax, cmap="BuPu", colorbar=False, text_kw={"fontsize": 13})
ax.set_title(f"KNN (k=5) 混淆矩阵 (Acc={results['KNN (k=5)']:.2%})", fontsize=13, fontweight="bold")

# 7c. MLP 不同架构的训练曲线对比
ax = fig7.add_subplot(2, 2, 3)
mlp_configs = [
    ("MLP (16,8)", (16, 8), 0.001),
    ("MLP (32,)", (32,), 0.001),
    ("MLP (64,32,16)", (64, 32, 16), 0.01),
]
for label, layers, alpha in mlp_configs:
    m = MLPClassifier(hidden_layer_sizes=layers, activation="relu", solver="adam",
                      alpha=alpha, max_iter=1000, random_state=42)
    m.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, m.predict(X_test_scaled))
    ax.plot(m.loss_curve_, label=f"{label} (Acc={acc:.2%})", linewidth=1.5)
ax.set_xlabel("迭代次数"); ax.set_ylabel("训练损失")
ax.set_title("不同 MLP 架构训练损失对比", fontsize=13, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 7d. 综合对比 - 交叉验证结果
ax = fig7.add_subplot(2, 2, 4)
cv_names = list(cv_results.keys())
cv_means = [cv_results[n][0] * 100 for n in cv_names]
cv_stds = [cv_results[n][1] * 100 for n in cv_names]
colors_cv = plt.cm.tab10(np.linspace(0, 1, len(cv_names)))
bars = ax.barh(cv_names, cv_means, xerr=cv_stds, color=colors_cv,
               edgecolor="white", capsize=5)
ax.set_xlabel("5折交叉验证准确率 (%)")
ax.set_title("各模型交叉验证结果", fontsize=13, fontweight="bold")
ax.set_xlim(0, 110)
for bar, mean, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{mean:.1f}%±{std:.1f}%", va="center", fontsize=8)
ax.grid(True, alpha=0.3, axis="x")

fig7.savefig("iris/output/07_knn_mlp_curves.svg", bbox_inches="tight")
plt.close(fig7)

# ============================================================
# 图8: 综合对比总结
# ============================================================
print("  [8/8] 综合总结图...")
fig8 = plt.figure(figsize=(22, 16))
fig8.suptitle("Iris 鸢尾花分类 — 8种方法综合对比", fontsize=18, fontweight="bold")

gs8 = GridSpec(4, 4, figure=fig8, hspace=0.5, wspace=0.4)

# 8a. 准确率柱状图
ax = fig8.add_subplot(gs8[0, :2])
method_names = list(results.keys())
method_accs = [results[n] * 100 for n in method_names]
colors_top = plt.cm.Set3(np.linspace(0, 1, len(method_names)))
bars = ax.barh(method_names, method_accs, color=colors_top, edgecolor="white", height=0.6)
ax.set_title("各方法测试准确率对比", fontsize=15, fontweight="bold")
ax.set_xlabel("准确率 (%)")
ax.set_xlim(0, 108)
for bar, acc in zip(bars, method_accs):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%", va="center", fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")

# 8b. 每类F1-Score对比
ax = fig8.add_subplot(gs8[0, 2:])
f1_scores = {}
for name in results:
    model = models.get(name, None)
    if name == "MLP (GridSearch最佳)":
        pred = y_pred_best_mlp
    elif model is not None:
        pred = model.predict(X_test_scaled)
    else:
        continue
    _, _, f1, _ = precision_recall_fscore_support(y_test, pred, zero_division=0)
    f1_scores[name] = f1
x = np.arange(3); w = 0.1
for idx, (name, f1s) in enumerate(f1_scores.items()):
    ax.bar(x + idx * w - w * (len(f1_scores) / 2), f1s, w, label=name, alpha=0.85, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(class_names)
ax.set_ylim(0, 1.15)
ax.set_title("各类别 F1-Score 对比", fontsize=13, fontweight="bold")
ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3, axis="y")

# 8c. ROC曲线 (OvR)
ax = fig8.add_subplot(gs8[1, :])
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
roc_models = {
    "逻辑回归": models["逻辑回归"],
    "SVM": models["SVM (RBF核)"],
    "随机森林": models["随机森林"],
    "KNN": models["KNN (k=5)"],
    "MLP最佳": best_mlp,
    "梯度提升树": models["梯度提升树"],
}
colors_roc = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#DDA0DD"]
for (name, model), color in zip(roc_models.items(), colors_roc):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_scaled)
        for i in range(3):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            if i == 0:
                ax.plot(fpr, tpr, color=color, linewidth=2,
                        label=f"{name} (AUC={roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("假正率 (FPR)"); ax.set_ylabel("真正率 (TPR)")
ax.set_title("ROC 曲线对比 (OvR, 以 setosa 类为例)", fontsize=14, fontweight="bold")
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

# 8d-g. 混淆矩阵一览
cm_list = []
for name in ["逻辑回归", "SVM (RBF核)", "随机森林", "KNN (k=5)"]:
    cm_list.append((name, confusion_matrix(y_test, models[name].predict(X_test_scaled))))
cm_list.append(("MLP最佳", confusion_matrix(y_test, y_pred_best_mlp)))
cm_list.append(("梯度提升树", confusion_matrix(y_test, models["梯度提升树"].predict(X_test_scaled))))
cm_list.append(("决策树", confusion_matrix(y_test, models["决策树"].predict(X_test_scaled))))
cm_list.append(("朴素贝叶斯", confusion_matrix(y_test, models["高斯朴素贝叶斯"].predict(X_test_scaled))))

for idx, (name, cm) in enumerate(cm_list):
    row = 2 if idx < 4 else 3
    col = idx % 4
    ax = fig8.add_subplot(gs8[row, col])
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="YlOrRd", colorbar=False, text_kw={"fontsize": 10},
        im_kw={"vmin": 0, "vmax": 15})
    ax.set_title(name, fontsize=11, fontweight="bold")

fig8.tight_layout()
fig8.savefig("iris/output/08_summary_advanced.svg", bbox_inches="tight")
plt.close(fig8)

print("  图表已保存至 iris/output/ 目录 (SVG 格式):")
print("    01_data_exploration_advanced.svg  — 数据探索")
print("    02_decision_boundaries_advanced.svg — 8模型决策边界")
print("    03_logistic_regression_advanced.svg — 逻辑回归分析")
print("    04_mlp_analysis.svg — MLP深入分析 (架构/正则/激活)")
print("    05_ensemble_methods.svg — 集成方法 (RF vs GBDT)")
print("    06_tree_nb_analysis.svg — 决策树 & 朴素贝叶斯")
print("    07_knn_mlp_curves.svg — KNN + MLP训练曲线")
print("    08_summary_advanced.svg — 综合对比总结")

print("\n完成！")
