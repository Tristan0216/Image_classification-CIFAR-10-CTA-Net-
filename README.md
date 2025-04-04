# Image_classification-CIFAR-10(CTA-Net & MyNet)
# 基于CIFAR-10数据集使用CTA-Net和CNN进行图像分类
### CTA-Net论文中并未给出代码，所以目前只能初步复现，后续还会再进行优化 </br>
训练```CTA-Net```实现分类任务，CTA-Net通过CNN分支提取局部特征，Transformer分支捕捉全局上下文信息，再通过特征聚合模块融合CNN和Transformer的特征，最后通过分类头输出分类结果。</br>
同时构建卷积神经网络并在相同数据集上进行训练，并比较与CTA-Net的训练结果</br>
cnn branch采用VGG-16作为特征提取网络
