import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 归一化（将像素值从 [0, 255] 缩放到 [0, 1]）
x_train, x_test = x_train / 255.0, x_test / 255.0

# 展开 28x28 图像到 1D 向量 (784 维)
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# 打印数据形状
print(f"Train data shape: {x_train.shape}, Test data shape: {x_test.shape}")

# 构建 Sequential 模型
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),  # 20% dropout，防止过拟合
    keras.layers.Dense(10, activation='softmax')
])


# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 打印模型结构
model.summary()

# 训练 5 轮（Epochs），批量大小 64
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# 训练 5 轮（Epochs），批量大小 64
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# 绘制训练 & 验证准确率
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training & Validation Accuracy')
plt.show()
