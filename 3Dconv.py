# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:59:45 2023

@author: jacky
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
os.chdir("C:/Users/jacky/OneDrive/桌面/s2ef_train_200K")
#%%
# 列出所有可用的 GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print("可用 GPU：", gpu)
else:
    print("沒有可用的 GPU")

#%%
data = np.load('system_models.npz')
datas = [data[key] for key in data.keys()]
for i, data in enumerate(datas):
    data = np.expand_dims(data, axis=0)
    datas[i] = np.expand_dims(data, axis=-1)

x_train = np.concatenate(datas, axis=0)
#%%
from system_labels import system_labels
y_train = system_labels
y_train = np.round(np.array(y_train)).astype(int)
#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

#%%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#%%
# 使用GPU訓練模型
from tensorflow.keras.callbacks import EarlyStopping
with tf.device('/device:GPU:0'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', input_shape=(69, 86, 197, 1)),
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
        tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
        tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    model.summary()
    history = model.fit(X_train, y_train, epochs=100, batch_size = 3, validation_data=(X_test, y_test), callbacks=[early_stop])

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


#%%
score = model.evaluate(X_test, y_test, batch_size = 1)
print('Test loss:', score[0])
print('Test MAE:', score[1])










