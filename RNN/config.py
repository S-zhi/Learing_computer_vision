class Config(object):
    # 定义一个配置类，用于存储模型的参数和文件路径

    # 指定存储古诗文本数据的文件路径
    poetry_file = 'poetry.txt'

    # 指定模型权重文件的路径
    weight_file = 'poetry_model.h5'

    # 定义了模型的最大输入序列长度，即前六个字
    max_len = 6

    # 定义训练时的批处理大小
    batch_size = 512

    # 定义学习率，用于模型训练的优化算法
    learning_rate = 0.001
