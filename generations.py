# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ##### 将Global_parameters这个模块中的参数以及函数导入该程序中

# %%
from Global_parameters import *

# %% [markdown]
# ##### 训练信道以及验证信道信息的获取

# %%
channel_train = np.load("./Hdatas/channel_train.npy")
train_size = channel_train.shape[0]
channel_validation = np.load("./Hdatas/channel_test.npy")
validation_size = channel_validation.shape[0]

# %% [markdown]
# ##### 训练数据的生成

# %%
def training_gen(bs, SNRdb=20):
    while True:
        input_samples_train = []
        input_labels_train = []
        for index_k in range(0, bs):
            data_bits = np.random.binomial(
                n=1, p=0.5, size=(payloadBits_per_OFDM,)
            )
            channel_response = channel_train[np.random.randint(0, train_size)]
            signal_output, para = ofdm_simulate(
                data_bits, channel_response, SNRdb
            )
            input_labels_train.append(data_bits[16:32])
            input_samples_train.append(signal_output)
        yield (np.asarray(input_samples_train), np.asarray(input_labels_train))


# %% [markdown]
# ##### 验证数据的生成

# %%
def validation_gen(bs, SNRdb=20):
    while True:
        input_samples_validation = []
        input_labels_validation = []
        for index_k in range(0, bs):
            data_bits = np.random.binomial(
                n=1, p=0.5, size=(payloadBits_per_OFDM,)
            )
            channel_response = channel_validation[
                np.random.randint(0, validation_size)
            ]
            signal_output, para = ofdm_simulate(
                data_bits, channel_response, SNRdb
            )
            input_labels_validation.append(data_bits[16:32])
            input_samples_validation.append(signal_output)
        yield (
            np.asarray(input_samples_validation),
            np.asarray(input_labels_validation),
        )
