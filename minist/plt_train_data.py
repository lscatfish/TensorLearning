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
    2.1467, 1.6280, 1.0654, 0.6474, 0.4089, 0.2869, 0.2237, 0.1894, 0.1647, 0.1471,
    0.1338, 0.1229, 0.1142, 0.1074, 0.1015, 0.0966, 0.0926, 0.0887, 0.0857, 0.0838,
    0.0813, 0.0795, 0.0779, 0.0767, 0.0759, 0.0752, 0.0748, 0.0745, 0.0743, 0.0708
]

# 训练准确率 (%)
train_acc = [
    36.82, 70.10, 79.57, 86.46, 90.66, 92.85, 94.10, 94.81, 95.42, 95.82,
    96.12, 96.48, 96.71, 96.91, 97.08, 97.22, 97.32, 97.46, 97.55, 97.62,
    97.66, 97.76, 97.81, 97.84, 97.85, 97.88, 97.88, 97.89, 97.90, 98.03
]

# 测试准确率 (%)
test_acc = [
    62.91, 76.06, 84.02, 89.72, 92.16, 93.58, 94.42, 95.18, 95.68, 96.10,
    96.23, 96.50, 96.53, 96.82, 96.91, 97.00, 97.17, 97.21, 97.19, 97.28,
    97.37, 97.36, 97.30, 97.33, 97.45, 97.39, 97.43, 97.40, 97.42, 97.52
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
