# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:03:44 2023

@author: jacky
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#%%
data = np.load('train_matrix_2k_3f.npz')
datas = [data[key] for key in data.keys()]
for i, data in enumerate(datas):
    data = np.expand_dims(data, axis=0)
    datas[i] = np.expand_dims(data, axis=-1)

x_train = np.concatenate(datas, axis=0)

#%%
label = []
with open('matrix_label_2k_3f.txt', 'r') as file:
    for line in file:
        label.append(line.strip())
#%%
from sklearn.model_selection import train_test_split
x_train = x_train[0]
X_train, X_test, y_train, y_test = train_test_split(x_train, label, test_size=0.1, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
#%% normalization
"""
mean = np.mean(X_train, axis = (0,1,2,3))
std = np.std(X_train, axis= (0,1,2,3))

X_train_norm = np.round((X_train - mean) / std, 5)
X_test_norm = np.round((X_test - mean) / std, 5)
"""

#%%
X_train = X_train[:,:,:,:,0]
X_val = X_val[:,:,:,:,0]
X_test = X_test[:,:,:,:,0]

y_train = list(map(float, y_train))
y_val = list(map(float, y_val))
y_test = list(map(float, y_test))
y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)
#%%
X_test_mean = np.mean(X_test, axis = (0, 1, 2))
X_test_std = np.std(X_test, axis = (0, 1, 2))
X_test_norm = (X_test - X_test_mean) / X_test_std

X_train_mean = np.mean(X_train, axis = (0, 1, 2))
X_train_std = np.std(X_train, axis = (0, 1, 2))
X_train_norm = (X_train - X_train_mean) / X_train_std

X_val_mean = np.mean(X_val, axis = (0, 1, 2))
X_val_std = np.std(X_val, axis = (0, 1, 2))
X_val_norm = (X_val - X_val_mean) / X_val_std

X_train = X_train_norm
X_test = X_test_norm
X_val = X_val_norm

del X_test_mean, X_test_std, X_test_norm
del X_train_mean, X_train_std, X_train_norm
del X_val_mean, X_val_std, X_val_norm
#%%
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Flatten, Dense

with tf.device('/device:GPU:0'):
    model = Sequential([
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(1028, activation='relu'),
        Dense(1028, activation='relu'),
        Dense(1028, activation='relu'),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.build(input_shape=(None, 214, 214, 3))  # 輸入形狀為 (batch_size, height, width, channels)
    model.compile(optimizer='adam', loss='mse', metrics=['mape'])
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    model.summary()
    history = model.fit(X_train, y_train, epochs=500, validation_data=(X_val, y_val), callbacks=[early_stop])

#%%
"""
# Plot training & validation loss values
fig = plt.figure(figsize=(8, 6), dpi=300)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# Plot training & validation accuracy values
fig = plt.figure(figsize=(8, 6), dpi=300)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
"""
#%%
score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test MAPE:', score[1])


# Get the predictions
y_pred = model.predict(X_test)
x = np.linspace(min(y_pred),max(y_pred))
y = x
plt.plot(x, y, color = 'red')
plt.scatter(y_test, y_pred)

plt.ylabel('prediction')
plt.xlabel('ground truth')
plt.show()


#%%
