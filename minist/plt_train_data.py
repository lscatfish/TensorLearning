import matplotlib.pyplot as plt
import numpy as np

# ===================== 绘图配置（论文高清、中文支持） =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  # 负号显示
plt.rcParams['figure.dpi'] = 300  # 300DPI高清分辨率
plt.rcParams['figure.figsize'] = (12, 8)  # 画布大小

# ===================== 手动录入你的训练数据（完全匹配你的日志） =====================
epochs = np.arange(1, 31)  # 1~30轮

# 训练损失
train_loss = [
    1.2775,
    0.3829,
    0.2408,
    0.1742,
    0.1349,
    0.1119,
    0.0960,
    0.0869,
    0.0787,
    0.0713,
    0.0675,
    0.0627,
    0.0596,
    0.0583,
    0.0544,
    0.0520,
    0.0509,
    0.0499,
    0.0488,
    0.0471,
    0.0461,
    0.0458,
    0.0435,
    0.0434,
    0.0441,
    0.0432,
    0.0441,
    0.0430,
    0.0420,
    0.0424,
]

# 训练准确率 (%)
train_acc = [
    64.14,
    88.27,
    92.78,
    94.73,
    95.95,
    96.52,
    97.08,
    97.36,
    97.53,
    97.81,
    97.91,
    98.09,
    98.09,
    98.16,
    98.30,
    98.34,
    98.47,
    98.36,
    98.51,
    98.48,
    98.56,
    98.55,
    98.61,
    98.64,
    98.64,
    98.64,
    98.64,
    98.70,
    98.69,
    98.69,
]

# 测试准确率 (%)
test_acc = [
    87.34,
    93.44,
    95.72,
    96.96,
    97.52,
    97.80,
    98.01,
    98.22,
    98.30,
    98.36,
    98.50,
    98.47,
    98.52,
    98.64,
    98.59,
    98.66,
    98.74,
    98.68,
    98.77,
    98.78,
    98.77,
    98.78,
    98.81,
    98.82,
    98.80,
    98.80,
    98.81,
    98.80,
    98.79,
    98.80,
]

# ===================== 绘制双子图：损失曲线 + 准确率曲线 =====================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 8))

# 子图1：训练损失曲线
ax1.plot(epochs, train_loss, 'b-o', linewidth = 2, markersize = 4, label = '训练损失')
ax1.set_title('模型训练损失变化曲线', fontsize = 14, pad = 10)
ax1.set_xlabel('训练轮次 (Epoch)', fontsize = 12)
ax1.set_ylabel('损失值 (Loss)', fontsize = 12)
ax1.legend(fontsize = 11)
ax1.grid(True, alpha = 0.3)
ax1.set_xticks(epochs[::2])  # 每隔2轮显示刻度

# 子图2：训练/测试准确率曲线
ax2.plot(epochs, train_acc, 'r-o', linewidth = 2, markersize = 4, label = '训练准确率')
ax2.plot(epochs, test_acc, 'g-o', linewidth = 2, markersize = 4, label = '测试准确率')
ax2.set_title('模型训练/测试准确率变化曲线', fontsize = 14, pad = 10)
ax2.set_xlabel('训练轮次 (Epoch)', fontsize = 12)
ax2.set_ylabel('准确率 (%)', fontsize = 12)
ax2.legend(fontsize = 11)
ax2.grid(True, alpha = 0.3)
ax2.set_xticks(epochs[::2])

# 调整布局，防止重叠
plt.tight_layout()

# 保存高清图片（直接用于论文）
plt.savefig('./训练过程曲线图.svg', bbox_inches = 'tight', format = "svg", transparent = False)
plt.close()

print("✅ 训练过程曲线图已生成！保存为：训练过程曲线图.svg")
print("📊 图表包含：训练损失曲线、训练/测试准确率对比曲线")
