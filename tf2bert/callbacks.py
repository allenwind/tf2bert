import os
import glob
import time
import datetime

import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping

class SaveBestModelOnMemory(tf.keras.callbacks.Callback):
    """训练期间, 每个 epoch 检查当前模型在验证集中是否是最好模型, 
    如果是, 则存放到内存中, 否则丢弃. 训练结束后, 使用最
    好的模型作为训练结果. 这种实现比 keras ModelCheckpoint
    更高效, 因为后者会把模型存储到磁盘上并带来较大的存储空间开销.
    如果要配合 EarlyStopping 使用, 需要合理的 stop 机制，否则模型
    并不是最优.
    """

    def __init__(self, monitor="val_loss", monitor_op=np.less, period=1, path=None):
        super(SaveBestModelOnMemory, self).__init__()
        self.monitor = monitor
        self.monitor_op = monitor_op

        # 每 period 个 epoch 进行一次 monitor_op,默认一次,这也是比较好理解的
        self.period = period
        # 如果存在，则保存最优模型到指定路径
        self.path = path
        # 记录最优权重的 epoch,用于调参时调整 epoch 以减少训练时间
        self.best_epochs = 0
        self.epochs_since_last_save = 0
        self.best = np.Inf
                
    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            current = logs.get(self.monitor, None)
            if current is None:
                print(self.monitor, "not support")
            else:
                # 满足条件更新权重
                if self.monitor_op(current, self.best):
                    # 记录当前的最优指标值
                    self.best = current
                    self.best_epochs = epoch + 1
                    # 更新最优权重
                    self.best_weights = self.model.get_weights()            
                    
    def on_train_end(self, logs=None):
        print("best epoch on", self.best_epochs)
        print("best loss", self.best)
        # 保存模型到指定路径
        if self.path is not None:
            self.model.save(self.path)
        
        # 训练结束时设置最优权重
        self.model.set_weights(self.best_weights)

class TimeStopping(tf.keras.callbacks.Callback):
    """计时停止"""

    def __init__(self, seconds=86400, verbose=0):
        super().__init__()
        self.seconds = seconds
        self.verbose = verbose
        self.stopped_epoch = None

    def on_train_begin(self, logs=None):
        self.stopping_time = time.time() + self.seconds

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if time.time() >= self.stopping_time:
            self.model.stop_training = True
            self.stopped_epoch = epoch

    def on_train_end(self, logs=None):
        if self.stopped_epoch is not None and self.verbose > 0:
            stopped_epoch = self.stopped_epoch + 1
            ftime = datetime.timedelta(seconds=self.seconds)
            print("stopping at epoch {} after training for {}".format(stopped_epoch, ftime))

    def get_config(self):
        base_configs = super().get_config()
        configs = {
            "seconds": self.seconds,
            "verbose": self.verbose
        }
        return dict(list(base_configs.items()) + list(configs.items()))
