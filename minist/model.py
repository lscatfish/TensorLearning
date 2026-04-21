import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

torch.set_float32_matmul_precision('medium')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DisableCompile = False


class MNIST_Split_Dataset(Dataset):
    def __init__(self, img_root_dir, train_mode = True, transform = None):
        """
        :param img_root_dir: 训练+测试图片共同根文件夹路径
        :param train_mode: True=加载训练集(training开头)，False=加载测试集(test开头)
        :param transform: 图像预处理
        """
        self.img_root = img_root_dir
        self.train_mode = train_mode
        self.transform = transform
        self.img_path_list = []  # 存储图片路径
        self.label_list = []  # 存储手写数字0-9标签

        # 遍历文件夹所有图片
        for img_name in os.listdir(self.img_root):
            # 只筛选JPG图片，大小写兼容
            if img_name.lower().endswith(('.jpg', '.jpeg')):
                # 判断是训练图还是测试图
                if self.train_mode:
                    if img_name.startswith("training"):
                        self.parse_name_label(img_name)
                else:
                    if img_name.startswith("test"):
                        self.parse_name_label(img_name)

    # 解析文件名：拆分下划线、提取标签
    def parse_name_label(self, img_name):
        # 去掉后缀.jpg，再按下划线分割
        name_no_suffix = img_name.split('.')[0]
        part1, idx, label_str = name_no_suffix.split('_')
        label = int(label_str)  # 手写数字0-9
        img_path = os.path.join(self.img_root, img_name)
        self.img_path_list.append(img_path)
        self.label_list.append(label)

    def __len__(self):
        # 数据集总图片数量
        return len(self.img_path_list)

    def __getitem__(self, index):
        # 读取图片 + 强制MNIST单通道灰度图
        img_path = self.img_path_list[index]
        label = self.label_list[index]
        img = Image.open(img_path).convert('L')  # L=单通道灰度，完美适配MNIST

        # 图像张量预处理
        if self.transform is not None:
            img = self.transform(img)

        return img, label


# 卷积化自注意力（ConvAttn2D [B, C, H, W]
@torch.compile(disable = DisableCompile)
class ConvAttn2D(nn.Module):
    """
    二维卷积化自注意力
    输入维度: [B, C, H, W]  批量, 通道, 高度, 宽度
    输出维度: [B, C, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        attn_channels: int = 16,
        dynamic_kernel_size: int = 3,  # 动态小卷积核 2D
        shared_large_kernel_size: int = 13,  # 共享大卷积核 2D
        use_residual: bool = True,
        use_norm: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.attn_channels = min(attn_channels, in_channels)
        self.dynamic_kernel_size = dynamic_kernel_size
        self.use_residual = use_residual
        self.use_norm = use_norm
        self.dyn_pad = dynamic_kernel_size // 2  # 2D 填充

        # 动态核生成器
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 2D 全局池化
        self.kernel_gen = nn.Sequential(
            nn.Conv2d(self.attn_channels, self.attn_channels, 1, bias = False),
            nn.ReLU(inplace = True),
            # 输出: 注意力通道 × 动态核高 × 动态核宽
            nn.Conv2d(self.attn_channels, self.attn_channels * dynamic_kernel_size * dynamic_kernel_size, 1, bias = False)
        )

        # 共享大核深度卷积（通道卷积，每个通道独立计算）
        self.shared_large_conv = nn.Conv2d(
            self.attn_channels, self.attn_channels,
            kernel_size = shared_large_kernel_size,
            padding = shared_large_kernel_size // 2,
            groups = self.attn_channels,  # 深度卷积
            bias = False,
        )

        # 特征融合 1x1 卷积
        self.fusion_conv = nn.Conv2d(self.attn_channels, in_channels, 1, bias = False)

        # 层归一化
        if self.use_norm:
            self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 截取注意力分支通道
        f_att = x[:, :self.attn_channels, :, :]  # [B, attn_C, H, W]

        # 2D 动态小核卷积
        # 生成动态卷积核
        kernel = self.kernel_gen(self.global_pool(f_att))
        kernel = kernel.view(B, self.attn_channels, self.dynamic_kernel_size, self.dynamic_kernel_size)

        # 维度重塑（适配 PyTorch 深度卷积）
        f_att_reshaped = f_att.reshape(1, B * self.attn_channels, H, W)
        kernel_reshaped = kernel.reshape(B * self.attn_channels, 1, self.dynamic_kernel_size, self.dynamic_kernel_size)

        # 0填充
        f_att_padded = F.pad(f_att_reshaped, (self.dyn_pad, self.dyn_pad, self.dyn_pad,
                                              self.dyn_pad), mode = 'constant', value = 0.0)

        # 动态深度卷积
        f_dyn = F.conv2d(
            f_att_padded,
            weight = kernel_reshaped,
            groups = B * self.attn_channels
        )
        f_dyn = f_dyn.reshape(B, self.attn_channels, H, W)

        # ==================== 共享大核卷积 ====================
        f_large = self.shared_large_conv(f_att)

        # ==================== 融合 + 残差 + 归一化 ====================
        f_att_fused = f_dyn + f_large
        out = self.fusion_conv(f_att_fused)

        # 残差连接
        if self.use_residual:
            out = out + x

        # 层归一化（适配 2D 维度）
        if self.use_norm:
            out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return out


# 分类主网络
@torch.compile(disable = DisableCompile)
class MNIST_ConvAttnNet(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        # 输入: [B, 1, 28, 28] MNIST 单通道灰度图

        # 浅层特征提取
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 1, ),
            nn.InstanceNorm2d(4, affine = True), nn.ReLU(inplace = True),
            nn.Conv2d(4, 8, kernel_size = 3, padding = 1),
            nn.InstanceNorm2d(8, affine = True), nn.ReLU(inplace = True),
            nn.Conv2d(8, 16, 3, 2, padding = 1),  # [B,16,14,14]
            nn.InstanceNorm2d(16, affine = True), nn.ReLU(inplace = True),
            nn.Conv2d(16, 16, kernel_size = 3, padding = 1),
            nn.InstanceNorm2d(16, affine = True), nn.ReLU(inplace = True),
            nn.Conv2d(16, 32, 3, 2, padding = 1),
            nn.InstanceNorm2d(32, affine = True), nn.ReLU(),  # [B,32,7,7]
            nn.Conv2d(32, 1, 1),
            nn.InstanceNorm2d(1, affine = True), nn.ReLU(inplace = True),
        )
        self.fc = nn.Sequential(
            nn.Linear(49, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Patch划分
@torch.compile(disable = DisableCompile)
class MNIST_PatchNet(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        # 输入: [B, 1, 28, 28] MNIST 单通道灰度图

        # 浅层特征提取
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 1, ),
            nn.InstanceNorm2d(8, affine = True), nn.GELU(),

            nn.Conv2d(8, 16, 2, 2, ),
            nn.InstanceNorm2d(16, affine = True), nn.ReLU(inplace = True), nn.Dropout(0.1),  # [B, 16, 14, 14]
            nn.Conv2d(16, 32, 3, 1, padding = 1, ),
            nn.InstanceNorm2d(32, affine = True), nn.ReLU(inplace = True), nn.Dropout(0.1),  # [B, 16, 14, 14]
            nn.Flatten(2),  # 展平到[B, 32, 196]

            nn.Conv1d(32, 64, 1),
            nn.InstanceNorm1d(64, affine = True), nn.ReLU(inplace = True), nn.Dropout(0.1),
            nn.Conv1d(64, 5, 1),
            nn.InstanceNorm1d(5, affine = True), nn.ReLU(inplace = True), nn.Dropout(0.1),
            nn.Flatten(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(980, 512),
            nn.ReLU(inplace = True), nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.fc(x)
        return x


# 数据预处理（MNIST 官方标准化）
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

IMG_FOLDER = r".\mnist_jpg"
train_dataset = MNIST_Split_Dataset(IMG_FOLDER, train_mode = True, transform = _transform)
test_dataset = MNIST_Split_Dataset(IMG_FOLDER, train_mode = False, transform = _transform)

BATCH_SIZE = 5000

# 4. 初始化模型、损失函数、优化器

if __name__ == "__main__":
    model = MNIST_PatchNet().to(DEVICE)
    EPOCHS = 30
    LR = 1e-3
    train_loader = DataLoader(train_dataset, BATCH_SIZE,
        shuffle = True, num_workers = 2, pin_memory = True, persistent_workers = True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE,
        shuffle = False, num_workers = 2, pin_memory = True, persistent_workers = True)
    # model.load_state_dict(torch.load(r'D:\code\TensorLearning\minist\md.pth', map_location = DEVICE))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)

    # 训练循环
    print(f"训练设备: {DEVICE}")
    print(f"训练集数量: {len(train_dataset)}, 测试集数量: {len(test_dataset)}")
    xxtest_acc = 0.9
    for epoch in range(EPOCHS):
        # 训练
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        train_acc = 100 * correct / total
        avg_loss = train_loss / len(train_loader)

        # 测试
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, pred = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (pred == labels).sum().item()
        sch.step()
        test_acc = 100 * test_correct / test_total
        if xxtest_acc < test_acc:
            xxtest_acc = test_acc
            torch.save(model.state_dict(), 'md_patch.pth')
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | 训练损失: {avg_loss:.4f} | 训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")
