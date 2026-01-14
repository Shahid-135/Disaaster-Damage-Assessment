import tensorflow as tf
from tensorflow.keras import layers

# -------------------------
# Uncertainty Weighted Fusion
# -------------------------
class UncertaintyWeightedFusion(layers.Layer):
    def __init__(
        self,
        units=128,
        mc_runs=3,
        dropout_rate=0.1,
        temperature=0.5,
        use_residual=True
    ):
        super().__init__()
        self.mc_runs = mc_runs
        self.temperature = tf.Variable(
            temperature, trainable=True, dtype=tf.float32
        )

        self.dense = layers.Dense(units, activation="relu")
        self.norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout_rate)
        self.post_proj = layers.Dense(units, activation="relu")
        self.use_residual = use_residual
        self.eps = 1e-6

    def call(self, cnn_feat, graph_feat, training=False):
        cnn_proj = self.norm(self.dense(cnn_feat))
        graph_proj = self.norm(self.dense(graph_feat))

        if training:
            cnn_mc = tf.stack([
                cnn_proj + self.dropout(cnn_proj, training=True)
                for _ in range(self.mc_runs)
            ])
            graph_mc = tf.stack([
                graph_proj + self.dropout(graph_proj, training=True)
                for _ in range(self.mc_runs)
            ])

            cnn_unc = tf.reduce_mean(
                tf.clip_by_value(tf.math.reduce_std(cnn_mc, 0), self.eps, 1e2),
                axis=1
            )
            graph_unc = tf.reduce_mean(
                tf.clip_by_value(tf.math.reduce_std(graph_mc, 0), self.eps, 1e2),
                axis=1
            )

            logits = tf.stack([-cnn_unc, -graph_unc], axis=1)
            weights = tf.nn.softmax(logits / self.temperature, axis=1)

            w_cnn = tf.expand_dims(weights[:, 0], -1)
            w_graph = tf.expand_dims(weights[:, 1], -1)
        else:
            w_cnn = w_graph = 0.5

        fused = w_cnn * cnn_proj + w_graph * graph_proj

        if self.use_residual:
            fused += 0.5 * (cnn_proj + graph_proj)

        fused = self.post_proj(fused)
        tf.debugging.check_numerics(fused, "NaNs in fusion output")
        return fused

