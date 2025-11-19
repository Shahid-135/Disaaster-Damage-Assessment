import tensorflow as tf
# -------------------------
# L1 regularizer (tiny wrapper)
# -------------------------
class L1Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, l1: float = 0.01):
        self.l1 = float(l1)

    def __call__(self, x):
        return self.l1 * tf.reduce_sum(tf.abs(x))

    def get_config(self) -> Dict:
        return {"l1": self.l1}


# -------------------------
# Tri-Axial Cross-Dimensional Attention
# -------------------------
class TriAxialCrossAttention(tf.keras.layers.Layer):
    """
    Tri-Axial Cross-Dimensional Attention.

    Args:
        reduction_ratio: intermediate channel reduction for global context MLP
        attn_dim: channel dimension used inside lightweight self-attention
        l1_lambda: coefficient for L1 regularizer applied to branch weights / convs
    """

    def __init__(self, reduction_ratio: int = 16, attn_dim: int = 8, l1_lambda: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = int(reduction_ratio)
        self.attn_dim = int(attn_dim)
        self.l1_lambda = float(l1_lambda)

        # placeholders for layers created in build()
        self._built = False

        # small convenience layers that don't depend on C
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.gc_fc1 = tf.keras.layers.Dense(self.reduction_ratio, activation="relu")
        self.gc_fc2 = tf.keras.layers.Dense(1, activation="sigmoid")

        # output conv for attention refinement (created in build to match input C if needed)
        self.out_conv = None

        # normalization (channel-wise)
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)

    def build(self, input_shape: Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]):
        # input_shape: (B, H, W, C)
        if len(input_shape) != 4:
            raise ValueError("Input tensor must be 4D: [B, H, W, C]")

        _, H, W, C = input_shape
        C = int(C)

        # --- Multi-scale convs for the three branches ---
        reg = L1Regularizer(l1=self.l1_lambda) if self.l1_lambda > 0.0 else None

        # Channel-branch transforms (operate on [B,1,1,C] pooled map -> output single attention scalar per channel)
        # Use Conv2D with 1 channel outputs so broadcasting is easy later.
        self.conv_c1 = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", kernel_regularizer=reg)
        self.conv_c3 = tf.keras.layers.SeparableConv2D(1, 3, padding="same", activation="sigmoid", kernel_regularizer=reg)
        self.conv_c_dil = tf.keras.layers.Conv2D(1, 3, padding="same", dilation_rate=2, activation="sigmoid", kernel_regularizer=reg)

        # Height-branch transforms (we will feed tensors of shape [B, H, 1, C])
        self.conv_h1 = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", kernel_regularizer=reg)
        self.conv_h3 = tf.keras.layers.SeparableConv2D(1, 3, padding="same", activation="sigmoid", kernel_regularizer=reg)
        self.conv_h_dil = tf.keras.layers.Conv2D(1, 3, padding="same", dilation_rate=2, activation="sigmoid", kernel_regularizer=reg)

        # Width-branch transforms (we will feed tensors of shape [B, 1, W, C])
        self.conv_w1 = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", kernel_regularizer=reg)
        self.conv_w3 = tf.keras.layers.SeparableConv2D(1, 3, padding="same", activation="sigmoid", kernel_regularizer=reg)
        self.conv_w_dil = tf.keras.layers.Conv2D(1, 3, padding="same", dilation_rate=2, activation="sigmoid", kernel_regularizer=reg)

        # Learnable branch weights (soft combination). Regularize with L1 optionally.
        self.branch_weights = self.add_weight(
            name="branch_weights",
            shape=(3,),
            initializer=tf.keras.initializers.Ones(),
            regularizer=L1Regularizer(l1=self.l1_lambda) if self.l1_lambda > 0.0 else None,
            trainable=True,
        )

        # Lightweight cross-dimensional attention convs (project to attn_dim)
        self.q_conv = tf.keras.layers.Conv2D(self.attn_dim, 1, kernel_regularizer=reg)
        self.k_conv = tf.keras.layers.Conv2D(self.attn_dim, 1, kernel_regularizer=reg)
        self.v_conv = tf.keras.layers.Conv2D(self.attn_dim, 1, kernel_regularizer=reg)

        # Final scalar map produced from attention output
        self.out_conv = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", kernel_regularizer=reg)

        self._built = True
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        x: tensor shape [B, H, W, C]
        returns: tensor shape [B, H, W, C]
        """

        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = tf.shape(x)[3]

        # -------------------------
        # Global context gating
        # -------------------------
        gc = self.global_pool(x)       # [B,1,1,C]
        gc = self.gc_fc1(gc)           # [B,1,1,reduction]
        gc = self.gc_fc2(gc)           # [B,1,1,1]
        gc = tf.broadcast_to(gc, tf.stack([B, H, W, 1]))  # [B,H,W,1]

        # -------------------------
        # Channel-axis attention branch
        # - pool over spatial dims HxW to get [B,1,1,C]
        # - transform with small convs -> scalar map broadcastable to [B,H,W,1] then multiplied over channels
        # -------------------------
        xc = tf.reduce_mean(x, axis=[1, 2], keepdims=True)  # [B,1,1,C]
        ca = (self.conv_c1(xc) + self.conv_c3(xc) + self.conv_c_dil(xc)) / 3.0  # [B,1,1,1]
        ca = tf.broadcast_to(ca, tf.stack([B, H, W, 1]))  # [B,H,W,1]
        ca_feat = x * ca  # broadcast over channel dim

        # -------------------------
        # Height-axis attention branch
        # - pool over width -> [B,H,1,C]
        # - apply convs (Conv2D accepts [B,H,1,C]) -> map to [B,H,1,1] -> broadcast to [B,H,W,C]
        # -------------------------
        xh = tf.reduce_mean(x, axis=2, keepdims=True)  # [B,H,1,C]
        ha = (self.conv_h1(xh) + self.conv_h3(xh) + self.conv_h_dil(xh)) / 3.0  # [B,H,1,1]
        ha = tf.broadcast_to(ha, tf.stack([B, H, W, 1]))  # [B,H,W,1]
        ha_feat = x * ha  # broadcast to channels

        # -------------------------
        # Width-axis attention branch
        # - pool over height -> [B,1,W,C]
        # - convs -> [B,1,W,1] -> broadcast to [B,H,W,C]
        # -------------------------
        xw = tf.reduce_mean(x, axis=1, keepdims=True)  # [B,1,W,C]
        wa = (self.conv_w1(xw) + self.conv_w3(xw) + self.conv_w_dil(xw)) / 3.0  # [B,1,W,1]
        wa = tf.broadcast_to(wa, tf.stack([B, H, W, 1]))
        wa_feat = x * wa

        # -------------------------
        # Fuse branches with learnable soft weights
        # -------------------------
        w = tf.nn.softmax(self.branch_weights)  # (3,)
        fused = w[0] * ca_feat + w[1] * ha_feat + w[2] * wa_feat  # [B,H,W,C]

        # -------------------------
        # Lightweight cross-dimensional self-attention
        # - project fused -> q,k,v (B,H,W,attn_dim)
        # - compute attention over spatial positions H*W
        # - produce attention scalar map (B,H,W,1)
        # -------------------------
        q = self.q_conv(fused)  # [B,H,W,attn_dim]
        k = self.k_conv(fused)
        v = self.v_conv(fused)

        # reshape to [B, H*W, attn_dim]
        q_flat = tf.reshape(q, [B, H * W, self.attn_dim])
        k_flat = tf.reshape(k, [B, H * W, self.attn_dim])
        v_flat = tf.reshape(v, [B, H * W, self.attn_dim])

        # scaled dot-product attention
        scores = tf.matmul(q_flat, k_flat, transpose_b=True) / tf.math.sqrt(tf.cast(self.attn_dim, tf.float32))  # [B, HW, HW]
        attn = tf.nn.softmax(scores, axis=-1)
        attn_out = tf.matmul(attn, v_flat)  # [B, HW, attn_dim]
        attn_out = tf.reshape(attn_out, [B, H, W, self.attn_dim])  # [B,H,W,attn_dim]

        att_map = self.out_conv(attn_out)  # [B,H,W,1]

        # -------------------------
        # Combine fused features, attention map, and global context
        # Residual + LayerNorm (channel-wise)
        # -------------------------
        out = fused * att_map * gc  # broadcasting multipliers
        out = self.norm(out + x)

        return out

    def get_config(self) -> Dict:
        cfg = super().get_config()
        cfg.update({"reduction_ratio": self.reduction_ratio, "attn_dim": self.attn_dim, "l1_lambda": self.l1_lambda})
        return cfg

