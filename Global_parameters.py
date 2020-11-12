#!/usr/bin/env python
# coding: utf-8

# ##### 程序的头文件

# In[ ]:


import os
import numpy as np
import tensorflow as tf


# ##### （OFDM）无线通信中一系列参数的设置

# In[ ]:


K = 64
CP = K//4
P = 64
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])

if P < K:
    pilotCarriers = allCarriers[::K//P]  # Pilots is every (K/P)th carrier.
    dataCarriers = np.delete(allCarriers, pilotCarriers)
else:  # K=P
    pilotCarriers = allCarriers
    dataCarriers = []

mu = 2
payloadBits_per_OFDM = K*mu
SNRdb = 20


# ##### 神经网络中参数的设置

# In[ ]:


n_hidden_1 = 500  # 1st layer num features
n_hidden_2 = 250  # 2nd layer num features
n_hidden_3 = 120  # 3rd layer num features
n_input = 256
n_output = 16  # every 16 bit are predicted by a model


# ##### QPSK调制函数的定义

# In[ ]:


def Modulation(bits):
    bit_r = bits.reshape((int(len(bits)/mu), mu))
    # This is just for QPSK or 4QAM modulation
    return (2*bit_r[:, 0]-1)+1j*(2*bit_r[:, 1]-1)


# ##### 64导频序列值的获取

# In[ ]:


Pilot_file_name = 'Pilot_' + str(P)
if os.path.isfile(Pilot_file_name):
    print('Load Training Pilots txt')
    # load file
    pilots_bits = np.loadtxt(Pilot_file_name, delimiter=',')
else:
    # write file
    pilots_bits = np.random.binomial(n=1, p=0.5, size=(K*mu, ))
    np.savetxt(Pilot_file_name, pilots_bits, delimiter=',')
pilotValue = Modulation(pilots_bits)


# ##### 逆快速傅里叶变换函数

# In[ ]:


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


# ##### 加循环前缀（CP）

# In[ ]:


def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


# ##### 信道模型函数的定义

# In[ ]:


def channel(signal, channelResponse, SNRdb):
    convolved = np.convolve(signal, channelResponse)  # 时域卷积
    signal_power = np.mean(abs(convolved**2))  # 信号功率
    sigma2 = signal_power * 10**(-SNRdb/10)  # 噪声功率
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape) +
                                 1j*np.random.randn(*convolved.shape))  # 高斯噪声
    return convolved + noise  # 时域接收信号


# ##### 去掉循环前缀

# In[ ]:


def removeCP(signal):
    return signal[CP:(CP+K)]


# ##### 快速傅里叶变换

# In[ ]:


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


# ##### 导频符号和原始发送符号共同经过一次模拟获得接收符号，该获得的接收符号作为训练输入

# In[ ]:


def ofdm_simulate(codeword, channelResponse, SNRdb):
    # 导频符号
    data_bits = np.random.binomial(n=1, p=0.5, size=(2*(K-P),))
    QAM = Modulation(data_bits)
    OFDM_data = np.zeros(K, dtype=complex)
    OFDM_data[pilotCarriers] = pilotValue
    OFDM_data[dataCarriers] = QAM
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX)
    OFDM_RX_noCP = DFT(OFDM_RX_noCP)

    # 发送信息符号
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword)
    symbol[np.arange(K)] = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = IDFT(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    OFDM_RX_noCP_codeword = DFT(OFDM_RX_noCP_codeword)

    # 获得的接收信号作为神经网络的输入
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword), np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse)


# ##### BER的定义

# In[ ]:


def bit_err(y_true, y_pred):
    err = 1 - tf.reduce_mean(tf.reduce_mean(tf.cast(tf.equal(
        tf.sign(y_pred - 0.5), tf.cast(tf.sign(y_true - 0.5), tf.float32)), dtype=tf.float32), 1))
    return err