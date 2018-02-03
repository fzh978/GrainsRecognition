# GrainsRecognition

>Tested on Python 2.7.12

Python 程序运行依赖opencv2.4.13、numpy、sklearn等

对预处理后的谷物图像提取形状、颜色、纹理等特征组成高维向量送入SVM训练并预测。SVM中RBF核下C和gamma参数直接影响最终分类性能。使用sklearn交叉训练选取最优值。

###python程序说明
- class_preprocess 进行图像预处理，包括灰度化、同一图像大小、图像滤波等
- class_shape提取形状特征，包括图像边缘提取、计算不变矩等
- class_rilbp提取纹理特征，使用旋转不变的局部二值化模式实现
- utils提取颜色特征，HSV模式下H分量
- class_features得到最终的组合特征。可以通过图像计算得到，也可以通过CSV文件读取先前计算好的数据
- class_csv包括读取csv文件信息和将计算好的特征数据保存起来，以免重复计算
- class_svm训练svm模型和预测输出结果
- rbf_parameter交叉训练选取最优参数

###运行结果
![](/result.png)

