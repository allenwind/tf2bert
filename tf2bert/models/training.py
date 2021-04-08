import tensorflow as tf

class AdversarialTraining(tf.keras.Model):
    """对抗训练器，像tf.keras.Model一样使用，
    这里实现的是Fast Gradient Method。"""

    def compile(self, optimizer, loss, metrics, eps=1.0, layer_name="embedding"):
        super(AdversarialTraining, self).compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        self.eps = eps
        self.layer_name = layer_name

    def train_step(self, data):
        embedding = self.get_layer(self.layer_name)
        embeddings = embedding.embeddings
        x, y = data
        with tf.GradientTape() as tape:
            tape.watch(embeddings)
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        # 计算Embedding梯度
        grads = tape.gradient(loss, embeddings)
        grads = tf.convert_to_tensor(grads)
        # 计算扰动
        delta = self.eps * grads / (norm(grads) + 1e-6)
        # 添加扰动到Embedding矩阵
        embeddings.assign_add(delta)
        # 执行普通的train_step
        results = super(AdversarialTraining, self).train_step(data)
        # 删除Embedding矩阵上的扰动
        embeddings.assign_sub(delta)
        return results
