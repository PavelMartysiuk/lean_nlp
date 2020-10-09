import os

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Films:
    def __init__(self, max_word_len, max_data_set_len):
        self.max_word_len = max_word_len
        self.max_data_set_len = max_data_set_len
        (self.x_train, self.y_train), (self.x_test, self.y_test) = imdb.load_data(num_words=self.max_word_len)

    @staticmethod
    def get_data_set_dict():
        word_dict = imdb.get_word_index()  # {word:code}
        reverse_word_dict = {}  # {code:word}
        for key, value in word_dict.items():
            reverse_word_dict[value] = key
        return reverse_word_dict

    def data_set_to_str(self, data_set):
        word_dict = self.get_data_set_dict()
        str_ = ''
        for digit in data_set:
            str_ += word_dict.get(digit) + ' '
        return str_

    def create_data_set_for_run(self, data_set):
        data_set = pad_sequences(data_set, maxlen=self.max_data_set_len)
        return data_set

    def create_network(self, input_data_set, output_data_set):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(self.max_data_set_len,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(input_data_set,
                            output_data_set,
                            epochs=25,
                            batch_size=128,
                            validation_split=0.1)
        return model, history


if __name__ == '__main__':
    obj1 = Films(10000, 200)
    input_data_set = obj1.create_data_set_for_run(obj1.x_train)
    model, history = obj1.create_network(input_data_set, obj1.y_train)
    plt.plot(history.history['accuracy'],
             label='Доля верных ответов на обучающем наборе')
    plt.plot(history.history['val_accuracy'],
             label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.legend()
    plt.show()
    data_set = obj1.create_data_set_for_run(obj1.x_train[:1])
    result = model.predict(data_set)
    print(result, np.argmax(result), sep='\n')
    print(obj1.data_set_to_str(obj1.x_train[0]))
