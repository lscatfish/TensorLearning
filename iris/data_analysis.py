"""
鸢尾花数据集 — 数据结构分析

生成统计量与可视化，用于报告中"数据探索与结构分析"章节：
1. 各类别特征均值/标准差
2. 特征间 Pearson 相关系数热力图
3. PCA 降维投影
4. 特征两两散点矩阵（pairplot）
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

os.makedirs("iris/output", exist_ok=True)

# ============================================================
# 0. 数据加载
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "iris.data")
data = np.loadtxt(DATA_PATH, delimiter=",", dtype=str)
X = data[:, :-1].astype(np.float64)
y = data[:, -1]

class_names = np.unique(y)
class_to_idx = {c: i for i, c in enumerate(class_names)}
y_labels = np.array([class_to_idx[c] for c in y])

feature_names = ["花萼长 (cm)", "花萼宽 (cm)", "花瓣长 (cm)", "花瓣宽 (cm)"]
feature_short = ["花萼长", "花萼宽", "花瓣长", "花瓣宽"]
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

# ============================================================
# 1. 各类别均值/标准差统计表
# ============================================================
print("=" * 60)
print("1. 各类别特征统计（均值 ± 标准差）")
print("=" * 60)

header = f"{'类别':　<14s}" + "".join(f"{fn:>16s}" for fn in feature_short)
print(f"\n{header}")
print("-" * len(header))

stats = {}
for i, cls in enumerate(class_names):
    mask = y_labels == i
    cls_data = X[mask]
    mean = cls_data.mean(axis=0)
    std = cls_data.std(axis=0, ddof=1)
    stats[cls] = (mean, std)
    row = f"{cls:　<14s}" + "".join(f"{mean[j]:>8.2f} ± {std[j]:<5.2f}" for j in range(4))
    print(row)

# 全局统计
global_mean = X.mean(axis=0)
global_std = X.std(axis=0, ddof=1)
print("-" * len(header))
row = f"{'全局':　<14s}" + "".join(f"{global_mean[j]:>8.2f} ± {global_std[j]:<5.2f}" for j in range(4))
print(row)

# ============================================================
# 2. 特征间 Pearson 相关系数
# ============================================================
print("\n" + "=" * 60)
print("2. 特征间 Pearson 相关系数矩阵")
print("=" * 60)

corr = np.corrcoef(X.T)
print(f"\n{'':>12s}" + "".join(f"{fn:>10s}" for fn in feature_short))
for i in range(4):
    row = f"{feature_short[i]:>10s}" + "".join(f"{corr[i, j]:>10.4f}" for j in range(4))
    print(row)

# 标注最强相关对
max_corr = 0
max_pair = None
for i in range(4):
    for j in range(i + 1, 4):
        if abs(corr[i, j]) > abs(max_corr):
            max_corr = corr[i, j]
            max_pair = (feature_short[i], feature_short[j])
print(f"\n  最强相关: {max_pair[0]} vs {max_pair[1]} = {max_corr:.4f}")

# ============================================================
# 3. PCA 分析
# ============================================================
print("\n" + "=" * 60)
print("3. PCA 降维分析")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_ratio = pca.explained_variance_ratio_
cumsum = np.cumsum(explained_ratio)

print(f"\n  各主成分方差解释率:")
for i, (er, cs) in enumerate(zip(explained_ratio, cumsum)):
    print(f"    PC{i + 1}: {er:.4f}（累计 {cs:.4f}）")
print(f"\n  前 2 主成分累计解释率: {cumsum[1]:.4f}")
print(f"  前 3 主成分累计解释率: {cumsum[2]:.4f}")

# PCA 成分载荷（各特征对主成分的贡献）
loadings = pca.components_
print(f"\n  主成分载荷（特征贡献）:")
for i in range(4):
    print(f"    PC1 载荷: {'  '.join(f'{loadings[0, j]:+.4f} ({feature_short[j]})' for j in range(4))}")
    break


# ============================================================
# 可视化
# ============================================================
print("\n生成可视化图表...")

# ============================================================
# 图1: 特征统计柱状图（均值 ± 标准差）
# ============================================================
print("  [1/4] 特征统计柱状图...")
fig1, axes = plt.subplots(2, 2, figsize=(14, 12))
fig1.suptitle("各类别在各特征上的均值 ± 标准差", fontsize=16, fontweight="bold")

for idx, (ax, fn) in enumerate(zip(axes.flatten(), feature_names)):
    x = np.arange(3)
    means = [stats[cls][0][idx] for cls in class_names]
    stds = [stats[cls][1][idx] for cls in class_names]
    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
                  capsize=6, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_ylabel(fn, fontsize=10)
    ax.set_title(f"{fn} 分布", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.05,
                f"{m:.2f}\n±{s:.2f}", ha="center", fontsize=8, fontweight="bold")

fig1.tight_layout()
fig1.savefig("iris/output/analysis_feature_stats.svg", bbox_inches="tight")
plt.close(fig1)

# ============================================================
# 图2: 特征相关系数热力图
# ============================================================
print("  [2/4] 相关系数热力图...")
fig2, ax = plt.subplots(figsize=(9, 8))
fig2.suptitle("特征间 Pearson 相关系数", fontsize=16, fontweight="bold", y=0.98)

im = ax.imshow(corr, cmap="RdBu_r", aspect="equal", vmin=-1, vmax=1)
ax.set_xticks(range(4))
ax.set_xticklabels(feature_short, rotation=25, ha="right", fontsize=11)
ax.set_yticks(range(4))
ax.set_yticklabels(feature_short, fontsize=11)

for i in range(4):
    for j in range(4):
        val = corr[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)

fig2.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
fig2.tight_layout()
fig2.savefig("iris/output/analysis_correlation_heatmap.svg", bbox_inches="tight")
plt.close(fig2)

# ============================================================
# 图3: PCA 降维投影
# ============================================================
print("  [3/4] PCA 降维投影...")
fig3 = plt.figure(figsize=(16, 7))
fig3.suptitle("PCA 降维分析", fontsize=16, fontweight="bold")

# 3a. 2D 投影散点图
ax = fig3.add_subplot(1, 2, 1)
for i, cls in enumerate(class_names):
    mask = y_labels == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], label=cls,
               alpha=0.8, edgecolors="k", linewidth=0.5, s=60)
ax.set_xlabel(f"PC1 ({explained_ratio[0]:.1%})", fontsize=11)
ax.set_ylabel(f"PC2 ({explained_ratio[1]:.1%})", fontsize=11)
ax.set_title("PCA 2D 投影（三类分布）", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 3b. 方差解释率 + 累计曲线
ax = fig3.add_subplot(1, 2, 2)
x = np.arange(1, 5)
bars = ax.bar(x, explained_ratio, color=colors[1] + "66" if len(colors) > 0 else "#4ECDC466",
              edgecolor="#4ECDC4", linewidth=1.5, label="单个主成分")
line = ax.plot(x, cumsum, "o-", color="#FF6B6B", linewidth=2, markersize=10,
               markeredgecolor="white", label="累计解释率")
ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="90% 阈值")
ax.set_xlabel("主成分", fontsize=11)
ax.set_xticks(x)
ax.set_ylabel("方差解释率", fontsize=11)
ax.set_title("主成分方差解释率", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

for i, (er, cs) in enumerate(zip(explained_ratio, cumsum)):
    ax.text(i + 1, er + 0.02, f"{er:.1%}", ha="center", fontsize=9, fontweight="bold")
    if i < 3:
        ax.text(i + 1 + 0.3, cs - 0.04, f"累计{cs:.1%}", fontsize=8,
                color="#FF6B6B", fontweight="bold")

fig3.tight_layout()
fig3.savefig("iris/output/analysis_pca_projection.svg", bbox_inches="tight")
plt.close(fig3)

# ============================================================
# 图4: 特征两两散点矩阵（pairplot）
# ============================================================
print("  [4/4] 特征两两散点矩阵...")
fig4, axes = plt.subplots(4, 4, figsize=(16, 14))
fig4.suptitle("特征两两散点矩阵", fontsize=16, fontweight="bold", y=0.96)

for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        if i == j:
            # 对角线：直方图
            for k, cls in enumerate(class_names):
                mask = y_labels == k
                ax.hist(X[mask, i], bins=12, alpha=0.5, color=colors[k], edgecolor="white")
            ax.set_title(feature_short[i], fontsize=9, fontweight="bold")
        else:
            # 非对角线：散点图
            for k, cls in enumerate(class_names):
                mask = y_labels == k
                ax.scatter(X[mask, j], X[mask, i], c=colors[k], label=cls,
                           alpha=0.6, edgecolors="k", linewidth=0.3, s=15)

        if i == 3:
            ax.set_xlabel(feature_short[j], fontsize=8)
        else:
            ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel(feature_short[i], fontsize=8)
        else:
            ax.set_yticklabels([])

# 图例放在右下角空位或单独位置
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                       markersize=8, label=cls) for c, cls in zip(colors, class_names)]
fig4.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, framealpha=0.8)

fig4.tight_layout(rect=[0, 0.03, 1, 0.95])
fig4.savefig("iris/output/analysis_feature_pairplot.svg", bbox_inches="tight")
plt.close(fig4)


print("\n已完成！SVG 图表保存在 iris/output/:")
print("  analysis_feature_stats.svg          — 各类别特征均值/标准差柱状图")
print("  analysis_correlation_heatmap.svg     — 特征相关系数热力图")
print("  analysis_pca_projection.svg          — PCA 降维投影 + 方差解释率")
print("  analysis_feature_pairplot.svg        — 特征两两散点矩阵")
