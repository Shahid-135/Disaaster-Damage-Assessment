import tensorflow as tf

# -------------------------
# Custom Contrastive Loss
# -------------------------
class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def call(self, z1, z2):
        z1 = tf.math.l2_normalize(z1, axis=1)
        z2 = tf.math.l2_normalize(z2, axis=1)

        sim = tf.matmul(z1, z2, transpose_b=True) / self.temperature
        batch_size = tf.shape(z1)[0]
        labels = tf.range(batch_size)

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=sim
            )
        )
        return loss
