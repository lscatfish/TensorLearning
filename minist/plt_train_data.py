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
    1.7002,
    0.5132,
    0.2229,
    0.1401,
    0.1030,
    0.0830,
    0.0709,
    0.0604,
    0.0536,
    0.0491,
    0.0439,
    0.0411,
    0.0374,
    0.0349,
    0.0327,
    0.0307,
    0.0293,
    0.0278,
    0.0267,
    0.0257,
    0.0250,
    0.0244,
    0.0237,
    0.0233,
    0.0229,
    0.0226,
    0.0224,
    0.0223,
    0.0222,
    0.0222,
]

# 训练准确率 (%)
train_acc = [
    58.20,
    87.46,
    93.46,
    95.88,
    96.99,
    97.52,
    97.92,
    98.21,
    98.44,
    98.57,
    98.75,
    98.82,
    98.93,
    99.00,
    99.09,
    99.12,
    99.18,
    99.23,
    99.25,
    99.31,
    99.34,
    99.35,
    99.36,
    99.39,
    99.40,
    99.41,
    99.41,
    99.42,
    99.42,
    99.42,
]

# 测试准确率 (%)
test_acc = [
    82.75,
    92.44,
    95.36,
    96.57,
    97.23,
    97.71,
    97.95,
    98.15,
    98.24,
    98.41,
    98.44,
    98.50,
    98.58,
    98.70,
    98.70,
    98.73,
    98.79,
    98.82,
    98.79,
    98.85,
    98.83,
    98.84,
    98.87,
    98.87,
    98.87,
    98.89,
    98.90,
    98.90,
    98.88,
    98.89,
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
