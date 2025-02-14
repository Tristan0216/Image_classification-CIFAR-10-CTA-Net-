import matplotlib.pyplot as plt

# 读取文件中的准确率数据
with open("accuracy.txt", "r") as file:
    accuracy_data = [line.strip() for line in file]

# 将准确率数据转换为浮点数
accuracy = [float(acc.replace("%", "")) for acc in accuracy_data]

# 绘制准确率曲线
epochs = range(1, len(accuracy) + 1)  # 训练轮数

plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracy, marker='', linestyle='-', color='red', label='Accuracy')

# 添加标题和标签
plt.title('Accuracy Curve', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)

# 添加网格
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

# 添加图例
plt.legend(loc='upper right', fontsize=10)

# 设置坐标轴范围
plt.ylim(0, 100)  # 纵轴范围
plt.xlim(0, len(accuracy) + 1)  # 横轴范围

# 设置刻度字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 显示图表
plt.show()