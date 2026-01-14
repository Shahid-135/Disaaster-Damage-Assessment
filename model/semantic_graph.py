import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from spektral.layers import GraphSageConv
from tensorflow.keras.layers import Dense

def build_similarity_adjacency(features, threshold=0.5):
    """
    features: np.ndarray (N, D)
    returns: scipy.sparse.csr_matrix (N, N)
    """
    sim = cosine_similarity(features)
    A = (sim > threshold).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    return sp.csr_matrix(A)


# ------------------------------------------------------------

class ClassAwareAdjacency(tf.keras.layers.Layer):
    def __init__(self, alpha=0.6):
        super().__init__()
        self.alpha = alpha

    def call(self, logits, adj):
        adj_dense = tf.sparse.to_dense(adj)

        probs = tf.nn.softmax(logits)
        class_sim = tf.matmul(probs, probs, transpose_b=True)
        class_sim = class_sim / tf.reduce_max(class_sim)

        refined = self.alpha * adj_dense + (1.0 - self.alpha) * class_sim
        return tf.sparse.from_dense(refined)


# ------------------------------------------------------------
class SemanticGraphBlock(tf.keras.layers.Layer):
    """
    Internal graph reasoning block.
    No loss. No standalone training. No final prediction.
    """

    def __init__(self, hidden_dim, num_classes, alpha=0.6, name="semantic_graph"):
        super().__init__(name=name)

        self.gcn1 = GraphSageConv(hidden_dim, activation="relu")
        self.logits_layer = Dense(num_classes)
        self.caar = ClassAwareAdjacency(alpha)
        self.gcn2 = GraphSageConv(hidden_dim, activation="relu")

    def call(self, inputs, training=False):
        """
        inputs: (x, a)
            x : Tensor (N, D)
            a : SparseTensor (N, N)
        """
        x, a = inputs

        h = self.gcn1([x, a])
        logits = self.logits_layer(h)

        if training:
            a_used = self.caar(logits, a)
        else:
            a_used = a

        h = self.gcn2([h, a_used])
        return h

