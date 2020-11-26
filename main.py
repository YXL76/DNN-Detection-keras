# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ##### 程序的头文件

# %%
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from generations import *

# %% [markdown]
# ##### 创建神经网络模型

# %%
model = models.Sequential()
model.add(
    layers.Dense(
        n_hidden_1, activation="relu", input_shape=(payloadBits_per_OFDM * 2,)
    )
)
model.add(layers.Dense(n_hidden_2, activation="relu"))
model.add(layers.Dense(n_hidden_3, activation="relu"))
model.add(layers.Dense(n_output, activation="sigmoid"))

# %% [markdown]
# ##### 配置优化器

# %%
model.compile(
    optimizer=optimizers.RMSprop(lr=0.001), loss="mse", metrics=[bit_err]
)

# %% [markdown]
# ##### 训练模型

# %%
history = model.fit(
    training_gen(1000, 20),
    steps_per_epoch=250,
    epochs=100,
    validation_data=validation_gen(1000, 20),
    validation_steps=1,
    verbose=2,
)

# %% [markdown]
# ##### 获得相应的数据，以及保存数据

# %%
loss = history.history["loss"]
bit_err = history.history["bit_err"]
val_loss = history.history["val_loss"]
val_bit_err = history.history["val_bit_err"]

np.savez(
    "./result/result.npz",
    loss=loss,
    bit_err=bit_err,
    val_loss=val_loss,
    val_bit_err=val_bit_err,
)

# %% [markdown]
# ##### 验证经训练得到的模型在各个SNR下的BER性能

# %%
BER = []
for SNR in range(5, 30, 2):
    y = model.evaluate_generator(
        validation_gen(10000, SNR), steps=1
    )  # y=[loss_value, metrics_value]
    BER.append(y[1])

np.save("./BER.npy", BER)

# %% [markdown]
# ##### 绘制训练ber和验证ber

# %%
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")

datas = np.load("./result/result.npz")
loss = datas["loss"]
bit_err = datas["bit_err"]
val_loss = datas["val_loss"]
val_bit_err = datas["val_bit_err"]

epochs = range(1, len(loss) + 1)

plt.semilogy(epochs, bit_err, "y", label="Training bit_err")
plt.semilogy(epochs, val_bit_err, "b", label="Validation bit_err")
plt.xlabel("epochs")
plt.ylabel("BER")
plt.title("Training and validation bit_err")
plt.legend()
plt.savefig("./result/bit_err.jpg")

plt.figure()

# %% [markdown]
# ##### 绘制训练损失和验证损失

# %%
plt.semilogy(epochs, loss, "y", label="Training loss")
plt.semilogy(epochs, val_loss, "b", label="Validation loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig("./result/loss.jpg")

plt.figure()

# %% [markdown]
# ##### 绘制基于深度学习的信号检测方法的BER性能

# %%
BER = np.load("./BER.npy")
SNR = range(5, 30, 2)

plt.semilogy(SNR, BER, "b", label="BER")
plt.xlabel("SNR(dB)")
plt.ylabel("BER")
plt.title("BER of deep learning based approach")
plt.legend()
plt.savefig("./result/BER.jpg")

plt.show()
