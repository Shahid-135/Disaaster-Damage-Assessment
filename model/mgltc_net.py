import tensorflow as tf
from tensorflow.keras import layers, Model

from fusion import UncertaintyWeightedFusion
from cross_modal import CrossModalGraphRefinement


def build_model(
    num_classes=3,
    input_shape_cnn=(49, 1024),
    input_shape_graph=(128,),
    handcraft_input_shape=(13,),
    exp_input_shape=(9,),
):
    raw_input = tf.keras.Input(shape=input_shape_cnn, name="raw_swin_input")
    graph_input = tf.keras.Input(shape=input_shape_graph, name="graphsage_input")
    handcraft_input = tf.keras.Input(shape=handcraft_input_shape)
    exp_input = tf.keras.Input(shape=exp_input_shape)

    # CNN branch
    x = layers.Reshape((7, 7, 1024))(raw_input)
    x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = TripletAttention()(x)
    cnn_feat = layers.GlobalAveragePooling2D()(x)
    cnn_proj = layers.Dense(128, activation="relu")(cnn_feat)

    # Graph branch
    graph_proj = layers.Dense(128, activation="relu")(graph_input)

    # Cross-modal refinement
    refined_graph = CrossModalGraphRefinement(128)(graph_proj, cnn_proj)

    # Fusion
    fused = UncertaintyWeightedFusion(128)(cnn_proj, refined_graph)

    # Handcrafted features
    h = layers.Dense(13, activation="relu")(layers.LayerNormalization()(handcraft_input))
    e = layers.Dense(9, activation="relu")(exp_input)
    aux = layers.Dense(22, activation="relu")(layers.Concatenate()([h, e]))

    # Classifier
    y = layers.Dense(128, activation="relu")(fused)
    y = layers.Dropout(0.2)(y)
    y = layers.Concatenate()([y, aux])
    y = layers.Dense(128, activation="relu")(y)
    y = layers.Dropout(0.2)(y)

    output = layers.Dense(num_classes, activation="softmax", name="classification_output")(y)

    cnn_norm = layers.Lambda(lambda x: tf.math.l2_normalize(x, 1))(cnn_proj)
    graph_norm = layers.Lambda(lambda x: tf.math.l2_normalize(x, 1))(fused)

    return Model(
        inputs=[graph_input, raw_input, handcraft_input, exp_input],
        outputs=[output, cnn_norm, graph_norm]
    )

