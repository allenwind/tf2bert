import tensorflow as tf

# 使用task_classification_bert_pooling.py 中的BERT作为教师网络
from task_classification_bert_pooling import model as teacher

"""
模型压缩技术可以分为两大类：
- 第一类，使用某种手段把大模型改为小模型：剪枝（Pruning）、量化（Quantization）
    - 剪枝，删减模型组件，使其变为较小的模型，但效果不至于变得太差。
    - 量化，不改变模型结构，权重改用其他的数值类型，但效果不至于变得太差。
- 第二类，蒸馏（Distillation），用一个小模型来学习大模型的输出
    - 使用更小的Transformer
    - 使用CNN或LSTM
这里提供蒸馏的实现。蒸馏的原理是，大模型的输出往往包括非常丰富的信息（logits），
而不是简单的one-hot，因此小模型能够从大模型的输出中学习更丰富的信息。
"""

class Distiller(tf.keras.Model):
    """蒸馏器"""

    def __init__(self, teacher, student, **kwargs):
        super(Distiller, self).__init__(**kwargs)
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss,
        distillation_loss,
        alpha=0.1,
        temperature=3):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss = student_loss
        self.distillation_loss = distillation_loss
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, inputs):
        X, y = inputs
        teacher_pred = self.teacher(X, training=False)
        with tf.GradientTape() as tape:
            student_pred = self.student(X, training=True)
            sloss = self.student_loss(y, student_pred)
            dloss = self.distillation_loss(
                tf.math.softmax(teacher_pred / self.temperature, axis=1),
                tf.math.softmax(student_pred / self.temperature, axis=1)
            )
            loss = self.alpha * sloss + (1 - self.alpha) * dloss

        student_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, student_vars)
        self.optimizer.apply_gradients(zip(gradients, student_vars))
        self.compiled_metrics.update_state(y, student_pred)
        results = {metric.name: metric.result() for metric in self.metrics}
        results.update({"student_loss": sloss, "distillation_loss": dloss})
        return results

    def test_step(self, inputs):
        X, y = inputs
        y_pred = self.student(X, training=False)
        sloss = self.student_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        results = {metric.name: metric.result() for metric in self.metrics}
        results.update({"student_loss": sloss})
        return results

def create_student_model():
    inputs = Input(shape=(maxlen,))
    x = Embedding(num_words, embedding_dims,
                  embeddings_initializer="glorot_normal",
                  mask_zero=True)(inputs)
    x = Dropout(0.1)(x)
    x = Conv1D(filters=128,
               kernel_size=3,
               padding="same",
               activation="relu",
               strides=1)(x)
    x = MaskedGlobalMaxPooling1D(hdims=128)(x)
    x = Dense(128)(x)
    x = Dropout(0.2)(x)
    x = Activation("relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model

# 创建学生网络
student = create_student_model()

# 训练教师网络
if not os.path.exists("teacher.weights"):
    teacher.fit(train_dataset, epochs=1, validation_data=valid_dataset)
    teacher.evaluate(valid_dataset)
    teacher.save_weights("teacher.weights")
else:
    teacher.save_weights("teacher.weights")

# 评估教师网络
teacher.evaluate(valid_dataset)

# 蒸馏
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
    student_loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    distillation_loss=tf.keras.losses.KLDivergence(),
    alpha=0.5,
    temperature=10
)
distiller.fit(train_dataset, epochs=3)
# 评估学生网络性能
distiller.evaluate(valid_dataset)
