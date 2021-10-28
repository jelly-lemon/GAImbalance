import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import *
from collections import Counter

from DataHelper import *
from MyMetrics import *





# 获取数据
x, y = get_data("Car")
class_num = len(Counter(y))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 创建模型
inputs = keras.Input(shape=x_train[0].shape)
flatten_1 = Flatten()(inputs)
dense_1 = Dense(20, activation="relu")(flatten_1)
outputs = Dense(class_num, activation="softmax")(dense_1)
model = keras.Model(inputs, outputs)
model.summary()
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer="Adam", metrics=["acc"])

print("x:", x_train.shape)
model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test))
# y_pred = model.predict(x_test)
# print(y_pred)