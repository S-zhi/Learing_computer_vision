import random
import os
from tensorflow import keras
import numpy as np
from data_utils import *
from keras.callbacks import LambdaCallback
from keras.models import Model, load_model
from keras.layers import LSTM, Dropout, Dense, Flatten, Bidirectional, Embedding, GRU, Input
from keras.optimizers import Adam

"""
    ______________________________________________________________________________
         本项目基于开源GitHub的开源项目
    1. author : S_zhi 
    2. E-mail : feng978744573@163.com
    3. project : 使用RNN模型来训练一个编写故事的模型，本模型仅仅是一个简单的故事生成模型，基于Keras框架，没有输入输出。
    4. 文件注释 : poetry.txt 为文件的训练的数据
    ______________________________________________________________________________

"""


# 文件的主函数 : 完成数据的训练
class Main_code(object):

    # 1.定义一个初始化函数，判断是否要进行模型的训练(1.1)
    def __init__(self, config):
        self.model = None
        self.do_train = True
        self.loaded_model = False
        self.config = config
        self.word2numF, self.num2word, self.words, self.files_content = preprocess_file(self.config)

        # 如果模型文件存在则直接加载模型，否则开始训练(本模型训练的时候不会对之前训练的模型接着训练，训练要一次完成)
        if os.path.exists(self.config.weight_file):

            self.model = load_model(self.config.weight_file)
            self.model.summary()
        else:
            self.train()
        self.do_train = False
        self.loaded_model = True

    # 2.建立RNN神经网络训练模型_建立模型(2.1)
    def build_model(self):
        input_tensor = Input(shape=(self.config.max_len,))
        embedd = Embedding(len(self.num2word) + 2, 300, input_length=self.config.max_len)(input_tensor)
        lstm = Bidirectional(GRU(128, return_sequences=True))(embedd)

        flatten = Flatten()(lstm)
        dense = Dense(len(self.words), activation='softmax')(flatten)
        self.model = Model(inputs=input_tensor, outputs=dense)
        optimizer = Adam(lr=self.config.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # 3.检验函数_设置值为temperature 来确定检验函数的正确性（3.1）
    def sample(self, preds, temperature=1.0):
        '''
        当temperature=1.0时，模型输出正常
        当temperature=0.5时，模型输出比较open
        当temperature=1.5时，模型输出比较保守
        在训练的过程中可以看到temperature不同，结果也不同
        '''
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    # 5. 检验函数_完成训练过程中返回每次模拟的情况（3.2）
    def generate_sample_result(self, epoch, logs):
        print("\n==================Epoch {}=====================".format(epoch))
        for diversity in [0.5, 1.0, 1.5]:
            print("------------Diversity {}--------------".format(diversity))
            start_index = random.randint(0, len(self.files_content) - self.config.max_len - 1)
            generated = ''
            sentence = self.files_content[start_index: start_index + self.config.max_len]
            generated += sentence
            for i in range(20):
                x_pred = np.zeros((1, self.config.max_len))
                for t, char in enumerate(sentence[-6:]):
                    x_pred[0, t] = self.word2numF(char)

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = self.num2word[next_index]

                generated += next_char
                sentence = sentence + next_char
            print(sentence)

    # 6. 拟合过程 ： 拟合1(4.1)
    def predict(self, text, temperature=1.0):

        if not self.loaded_model:
            return
        # 其余部分保持不变

        with open(self.config.poetry_file, 'r', encoding='utf-8') as f:
            file_list = f.readlines()
        random_line = random.choice(file_list)
        # 如果给的text不到四个字，则随机补全
        if not text or len(text) != 4:
            for _ in range(4 - len(text)):
                random_str_index = random.randrange(0, len(self.words))
                text += self.num2word.get(random_str_index) if self.num2word.get(random_str_index) not in [',', '。',
                                                                                                           '，'] else self.num2word.get(
                    random_str_index + 1)
        seed = random_line[-(self.config.max_len):-1]

        res = ''

        seed = 'c' + seed

        for c in text:
            seed = seed[1:] + c
            for j in range(5):
                x_pred = np.zeros((1, self.config.max_len))
                for t, char in enumerate(seed):
                    x_pred[0, t] = self.word2numF(char)

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, 1.0)
                next_char = self.num2word[next_index]
                seed = seed[1:] + next_char
            res += seed
        return res

    # 7. 拟合过程 ： 拟合2(4.2)
    def data_generator(self):
        '''生成器生成数据'''
        i = 0
        while 1:
            x = self.files_content[i: i + self.config.max_len]
            y = self.files_content[i + self.config.max_len]

            puncs = [']', '[', '（', '）', '{', '}', '：', '《', '》', ':']
            if len([i for i in puncs if i in x]) != 0:
                i += 1
                continue
            if len([i for i in puncs if i in y]) != 0:
                i += 1
                continue

            y_vec = np.zeros(
                shape=(1, len(self.words)),
                dtype=bool
            )
            y_vec[0, self.word2numF(y)] = 1.0

            x_vec = np.zeros(
                shape=(1, self.config.max_len),
                dtype=np.int32
            )

            for t, char in enumerate(x):
                x_vec[0, t] = self.word2numF(char)
            yield x_vec, y_vec
            i += 1

    # 4. 训练模型，训练模型的第二阶段_完成模型的建立(2.2)
    def train(self):

        number_of_epoch = len(self.files_content) // self.config.batch_size

        if not self.model:
            self.build_model()

        self.model.summary()

        self.model.fit_generator(
            generator=self.data_generator(),
            verbose=True,
            steps_per_epoch=self.config.batch_size,
            epochs=number_of_epoch,
            callbacks=[
                keras.callbacks.ModelCheckpoint(self.config.weight_file, save_weights_only=False),
                LambdaCallback(on_epoch_end=self.generate_sample_result)
            ]
        )


if __name__ == '__main__':
    from config import Config

    # 导入config类：
    model = Main_code(Config)
    while 1:
        text = ""
        sentence = model.predict(text)
        sentence_normal = model.predict(text, temperature=1.0)  # 正常风格
        sentence_open = model.predict(text, temperature=0.5)  # 比较开放的风格
        sentence_conservative = model.predict(text, temperature=1.5)  # 保守的风格
        print(sentence)
