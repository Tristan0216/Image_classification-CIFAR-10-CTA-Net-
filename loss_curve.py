import matplotlib.pyplot as plt

# 读取文件中的平均损失数据
with open("averageloss.txt", "r") as file:
    loss_data = [float(line.strip()) for line in file]

# 绘制损失曲线
epochs = range(1, len(loss_data) + 1)  # 训练轮数

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_data, marker='', linestyle='-', color='red', label='Loss')

# 添加标题和标签
plt.title('Loss Curve', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)

# 添加网格
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

# 添加图例
plt.legend(loc='upper right', fontsize=10)

# 设置坐标轴范围
plt.ylim(min(loss_data) - 0.1, max(loss_data) + 0.1)  # 纵轴范围
plt.xlim(0, len(loss_data) + 1)  # 横轴范围

# 设置刻度字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 显示图表
plt.show()