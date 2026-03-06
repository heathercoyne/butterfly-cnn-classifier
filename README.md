ButterflyNet开发
    本项目实现了一个基于自定义 CNN 的蝴蝶图像分类器。最终可运行脚本为 train2.2.py

环境与依赖
    Python 3.13
    安装依赖:
    pip install torch torchvision numpy pillow scikit-learn matplotlib tqdm

可自定义参数（命令行）

    以下参数均可在终端运行时自定义：
    --img_size 输入图像大小，必须是 16 的倍数
    --batch_size 批大小
    --epochs 训练轮数
    --lr 学习率
    --momentum SGD 动量
    --weight_decay L2 正则强度
    --dropout 分类头中的 dropout 比例
    --strong_aug 是否开启更强的数据增强
    --use_pool 是否启用池化进行下采样
    --pool_type 池化类型，max 或 avg

默认参数:
    img_size=128, batch_size=32, epochs=80, lr=0.01,
    momentum=0.9, weight_decay=5e-4, dropout=0.2,
    pool_type=max（仅当 --use_pool 时生效）

复现当前最好效果的推荐命令：
python -u train2.2.py --strong_aug --use_pool --momentum 0.8

本项目使用 torchvision.datasets.ImageFolder 读取数据。目录需类似：
ButterflyClassificationDataset/
├─ class_A/
│  ├─ img001.jpg
│  ├─ ...
├─ class_B/
│  ├─ img101.jpg
│  ├─ ...
└─ ...
指定数据集路径的方式：
python -u train2.2.py --data_dir ./ButterflyClassificationDataset


输出与结果文件
    运行后会在项目根目录生成 output/ 文件夹，包含：
    - best.pt：验证集上表现最好的模型权重
    - run_config.json：本次训练的参数与环境配置
    - metrics.txt：测试集总体指标
    - per_class_metrics.csv：各类别准确率
    - confusion_matrix.png：混淆矩阵可视化
