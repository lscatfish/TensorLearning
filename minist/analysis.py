# 分析模型
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, \
    roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from minist.model import (MNIST_ConvAttnNet, _transform, DEVICE,  IMG_FOLDER,
                          test_dataset, BATCH_SIZE, train_dataset,
    MNIST_PatchNet)
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']  # 备选字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

NUM_CLASSES = 10


def load_trained_model():
    """加载训练好的模型"""
    model = MNIST_PatchNet().to(DEVICE)
    # 加载你训练好的权重
    model.load_state_dict(torch.load('./md_patch.pth', map_location = DEVICE))
    model.eval()
    return model


@torch.no_grad()
def predict_all(loader):
    """核心：批量预测（获取标签/概率）"""
    y_true = []
    y_pred = []
    y_score = []  # softmax概率，用于ROC/PR曲线
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim = 1)
        probs = F.softmax(outputs, dim = 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_score.extend(probs.cpu().numpy())
    return np.array(y_true), np.array(y_pred), np.array(y_score)


def calculate_metrics(y_true, y_pred):
    """计算核心评价指标"""
    # 总体准确率
    acc = accuracy_score(y_true, y_pred)
    # 精确率、召回率、F1（宏平均/加权平均）
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average = 'macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average = 'weighted')
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # 分类报告（每类指标）
    report = classification_report(y_true, y_pred, digits = 4)

    # 打印格式化指标
    print("=" * 50)
    print("          模型综合评价指标")
    print("=" * 50)
    print(f"总体准确率 (Accuracy):\t{acc:.4f} ({acc * 100:.2f}%)")
    print(f"宏平均精确率 (Precision):\t{precision_macro:.4f}")
    print(f"宏平均召回率 (Recall):\t{recall_macro:.4f}")
    print(f"宏平均F1分数 (F1):\t{f1_macro:.4f}")
    print(f"加权F1分数 (F1):\t{f1_weighted:.4f}")
    print("=" * 50)
    print("\n每类详细分类报告：")
    print(report)
    return acc, cm


def plot_confusion_matrix(cm, classes = range(10)):
    """绘制混淆矩阵热力图"""
    plt.figure(figsize = (10, 8))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',
        xticklabels = classes, yticklabels = classes)
    plt.title(f'模型混淆矩阵 (准确率: {acc * 100:.2f}%)', fontsize = 14, pad = 20)
    plt.xlabel('预测标签', fontsize = 12)
    plt.ylabel('真实标签', fontsize = 12)
    plt.tight_layout()
    plt.savefig('./混淆矩阵.svg', bbox_inches = 'tight', format = "svg", transparent = False)
    plt.close()
    print("✅ 混淆矩阵已保存：混淆矩阵.svg")


def plot_roc_curve(y_true_bin, y_score):
    """绘制多分类 ROC 曲线"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 宏平均ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= NUM_CLASSES
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize = (10, 8))
    plt.plot(fpr["macro"], tpr["macro"], label = f'宏平均ROC曲线 (AUC = {roc_auc["macro"]:.4f})', linewidth = 2)
    for i in range(NUM_CLASSES):
        plt.plot(fpr[i], tpr[i], linestyle = '--', label = f'类别 {i} (AUC = {roc_auc[i]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('MNIST 模型 ROC 曲线', fontsize = 14)
    plt.legend(loc = 'lower right')
    plt.tight_layout()
    plt.savefig('./ROC曲线.svg', bbox_inches = 'tight', format = "svg", transparent = False)
    plt.close()
    print("✅ ROC曲线已保存：ROC曲线.svg")


def plot_pr_curve(y_true_bin, y_score):
    """绘制多分类 PR 曲线"""
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(NUM_CLASSES):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    plt.figure(figsize = (10, 8))
    for i in range(NUM_CLASSES):
        plt.plot(recall[i], precision[i], label = f'类别 {i} (AUC = {pr_auc[i]:.4f})')
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('MNIST 模型 PR 曲线', fontsize = 14)
    plt.legend(loc = 'lower left')
    plt.tight_layout()
    plt.savefig('./PR曲线.svg', bbox_inches = 'tight', format = "svg", transparent = False)
    plt.close()
    print("✅ PR曲线已保存：PR曲线.svg")


@torch.no_grad()
def get_dataset_metrics(loader, model):
    """绘制训练/测试准确率&损失曲线"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, acc


def plot_acc_loss_curve():
    # 单值柱状图（展示最终训练/测试结果）
    x = ['训练集', '测试集']
    accs = [train_acc, test_acc]
    losses = [train_loss, test_loss]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))
    # 准确率
    ax1.bar(x, accs, color = ['#4CAF50', '#2196F3'])
    ax1.set_title('模型准确率对比', fontsize = 12)
    ax1.set_ylabel('准确率')
    for i, v in enumerate(accs):
        ax1.text(i, v + 0.01, f'{v * 100:.2f}%', ha = 'center')

    # 损失
    ax2.bar(x, losses, color = ['#FF9800', '#F44336'])
    ax2.set_title('模型损失对比', fontsize = 12)
    ax2.set_ylabel('损失值')
    for i, kk in enumerate(losses):
        ax2.text(i, kk + 0.0002, f'{kk:.4f}', ha = 'center')

    plt.tight_layout()
    plt.savefig('./准确率损失对比.svg', bbox_inches = 'tight', format = "svg", transparent = False)
    plt.close()
    print("✅ 准确率损失对比图已保存：准确率损失对比.svg")


if __name__ == '__main__':
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle = False, num_workers = 1, pin_memory = True)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle = False, num_workers = 1, pin_memory = True)
    model = load_trained_model()

    # 获取测试集全部预测结果
    y_true, y_pred, y_score = predict_all(test_loader)
    # 二值化标签（用于多分类ROC/PR）
    y_true_bin = label_binarize(y_true, classes = range(NUM_CLASSES))

    acc, cm = calculate_metrics(y_true, y_pred)

    plot_confusion_matrix(cm)
    plot_roc_curve(y_true_bin, y_score)
    plot_pr_curve(y_true_bin, y_score)

    # 计算训练集/测试集指标
    train_loss, train_acc = get_dataset_metrics(train_loader, model)
    test_loss, test_acc = get_dataset_metrics(test_loader, model)

    plot_acc_loss_curve()

    print("\n🎉 模型分析完成！所有图表和指标已生成，可直接用于论文！")
