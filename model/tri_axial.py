
import tensorflow as tf

# ------------------------------------------------------------
# L1 Regularizer
# ------------------------------------------------------------
class L1Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, l1=0.01):
        self.l1 = l1

    def __call__(self, x):
        return self.l1 * tf.reduce_sum(tf.abs(x))

    def get_config(self):
        return {"l1": float(self.l1)}


# ------------------------------------------------------------
# Triplet Attention Layer
# ------------------------------------------------------------
class TripletAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        reduction_ratio=16,
        attention_dim=8,
        l1_lambda=0.01,
        name="triplet_attention",
    ):
        super().__init__(name=name)

        # Channel attention convolutions
        self.conv_c1 = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")
        self.conv_c3 = tf.keras.layers.SeparableConv2D(1, 3, padding="same", activation="sigmoid")
        self.conv_c_dilated = tf.keras.layers.Conv2D(
            1, 3, padding="same", dilation_rate=2, activation="sigmoid"
        )

        # Height attention convolutions
        self.conv_h1 = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")
        self.conv_h3 = tf.keras.layers.SeparableConv2D(1, 3, padding="same", activation="sigmoid")
        self.conv_h_dilated = tf.keras.layers.Conv2D(
            1, 3, padding="same", dilation_rate=2, activation="sigmoid"
        )

        # Width attention convolutions
        self.conv_w1 = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")
        self.conv_w3 = tf.keras.layers.SeparableConv2D(1, 3, padding="same", activation="sigmoid")
        self.conv_w_dilated = tf.keras.layers.Conv2D(
            1, 3, padding="same", dilation_rate=2, activation="sigmoid"
        )

        # Learnable branch weights with L1 regularization
        self.branch_weights = self.add_weight(
            name="branch_weights",
            shape=(3,),
            initializer="uniform",
            regularizer=L1Regularizer(l1=l1_lambda),
            trainable=True,
        )

        # Global context
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.global_dense = tf.keras.layers.Dense(reduction_ratio, activation="relu")
        self.global_dense_restore = tf.keras.layers.Dense(1, activation="sigmoid")

        # Lightweight self-attention
        self.attention_dim = attention_dim
        self.query_conv = tf.keras.layers.Conv2D(
            attention_dim, 1, kernel_regularizer=L1Regularizer(l1=l1_lambda)
        )
        self.key_conv = tf.keras.layers.Conv2D(
            attention_dim, 1, kernel_regularizer=L1Regularizer(l1=l1_lambda)
        )
        self.value_conv = tf.keras.layers.Conv2D(
            attention_dim, 1, kernel_regularizer=L1Regularizer(l1=l1_lambda)
        )
        self.out_conv = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")

        self.norm = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def call(self, x):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # Global context
        global_ctx = self.global_pool(x)
        global_ctx = self.global_dense(global_ctx)
        global_ctx = self.global_dense_restore(global_ctx)
        global_ctx = tf.tile(global_ctx, [1, H, W, 1])

        # ---------------- Channel Attention ----------------
        x_c = tf.transpose(x, [0, 3, 1, 2])
        ca = tf.reduce_mean(x_c, axis=[2, 3], keepdims=True)
        ca = tf.transpose(ca, [0, 2, 3, 1])

        ca = (self.conv_c1(ca) + self.conv_c3(ca) + self.conv_c_dilated(ca)) / 3.0
        ca = tf.transpose(ca, [0, 3, 1, 2])
        x_c = tf.transpose(x_c * ca, [0, 2, 3, 1])

        # ---------------- Height Attention ----------------
        x_h = tf.transpose(x, [0, 2, 1, 3])
        ha = tf.reduce_mean(x_h, axis=1, keepdims=True)
        ha = (self.conv_h1(ha) + self.conv_h3(ha) + self.conv_h_dilated(ha)) / 3.0
        ha = tf.transpose(ha, [0, 2, 1, 3])
        x_h = x * ha

        # ---------------- Width Attention ----------------
        wa = tf.reduce_mean(x, axis=1, keepdims=True)
        wa = (self.conv_w1(wa) + self.conv_w3(wa) + self.conv_w_dilated(wa)) / 3.0
        wa = tf.tile(wa, [1, H, 1, C])
        x_w = x * wa

        # Weighted fusion of branches
        weights = tf.nn.softmax(self.branch_weights)
        combined = weights[0] * x_c + weights[1] * x_h + weights[2] * x_w

        # ---------------- Cross-Dimensional Self-Attention ----------------
        ca_map = tf.reduce_mean(x_c, axis=-1, keepdims=True)
        ha_map = tf.reduce_mean(x_h, axis=-1, keepdims=True)
        wa_map = tf.reduce_mean(x_w, axis=-1, keepdims=True)

        attn_maps = tf.concat([ca_map, ha_map, wa_map], axis=-1)
        attn_maps = tf.reshape(attn_maps, [B, H * W, 3])
        attn_maps = tf.reshape(attn_maps, [B, H, W, 3])

        Q = self.query_conv(attn_maps)
        K = self.key_conv(attn_maps)
        V = self.value_conv(attn_maps)

        Q = tf.reshape(Q, [B, H * W, self.attention_dim])
        K = tf.reshape(K, [B, H * W, self.attention_dim])
        V = tf.reshape(V, [B, H * W, self.attention_dim])

        scores = tf.matmul(Q, K, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(self.attention_dim, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)

        attn_out = tf.matmul(weights, V)
        attn_out = tf.reshape(attn_out, [B, H, W, self.attention_dim])
        cross_attn = self.out_conv(attn_out)

        # Final output
        out = combined * cross_attn * global_ctx
        out = self.norm(out + x)

        return out
