#!/usr/bin/env python
# coding: utf-8


import os
import torch 
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import tensorflow as tf
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,classification_report,precision_recall_fscore_support, 
confusion_matrix, roc_curve, auc, precision_recall_curve)


# Labels
label_mapping = {"severe_damage": 0, "mild_damage": 1, "little_or_no_damage": 2}
df_train = pd.read_table("/home/shahid/000.Damage Severity/aman/embeddings/task_damage_text_img_train.tsv")
df_val = pd.read_table("/home/shahid/000.Damage Severity/aman/embeddings/task_damage_text_img_dev.tsv")
df_test = pd.read_table("/home/shahid/000.Damage Severity/aman/embeddings/task_damage_text_img_test.tsv")
df_train["damage_label"] = df_train["label"].map(label_mapping)
df_val["damage_label"] = df_val["label"].map(label_mapping)
df_test["damage_label"] = df_test["label"].map(label_mapping)


# Print counts for each split
print("Train label counts:")
print(df_train["damage_label"].value_counts())

print("\nValidation label counts:")
print(df_val["damage_label"].value_counts())

print("\nTest label counts:")
print(df_test["damage_label"].value_counts())


for name, df in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
    print(f"\n{name} label counts:")
    counts = df["label"].value_counts()
    for lbl in label_mapping:
        print(f"{lbl}: {counts.get(lbl, 0)}")



# ----- Load Image Features and Labels -----
train_image_raw = np.load("/home/shahid/000.Damage Severity/aman/embeddings/CrisisMMD/Swin/train_feature_last_hidden_state.npy", allow_pickle=True)
val_image_raw = np.load("/home/shahid/000.Damage Severity/aman/embeddings/CrisisMMD/Swin/val_feature_last_hidden_state.npy", allow_pickle=True)
test_image_raw = np.load("/home/shahid/000.Damage Severity/aman/embeddings/CrisisMMD/Swin/test_feature_last_hidden_state.npy", allow_pickle=True)

graph_img_train = np.load("train_hidden_features.npy", allow_pickle=True)
graph_img_val = np.load("val_hidden_features.npy", allow_pickle=True)
graph_img_test = np.load("test_hidden_features.npy", allow_pickle=True)

handcraft_img_train = np.load("/home/shahid/000.Damage Severity/aman/embeddings/1.Final_model/1.final_model/novel_features/train_handcrafted_features.npy", allow_pickle=True)
handcraft_img_val = np.load("/home/shahid/000.Damage Severity/aman/embeddings/1.Final_model/1.final_model/novel_features/val_handcrafted_features.npy", allow_pickle=True)
handcraft_img_test = np.load("/home/shahid/000.Damage Severity/aman/embeddings/1.Final_model/1.final_model/novel_features/test_handcrafted_features.npy", allow_pickle=True)

exp_img_train = np.load("/home/shahid/000.Damage Severity/aman/embeddings/1.Final_model/1.final_model/explainability/train_exp_features.npy", allow_pickle=True)
exp_img_val = np.load("/home/shahid/000.Damage Severity/aman/embeddings/1.Final_model/1.final_model/explainability/val_exp_features.npy", allow_pickle=True)
exp_img_test = np.load("/home/shahid/000.Damage Severity/aman/embeddings/1.Final_model/1.final_model/explainability/test_exp_features.npy", allow_pickle=True)

# Labels
label_mapping = {"severe_damage": 0, "mild_damage": 1, "little_or_no_damage": 2}
df_train = pd.read_table("/home/shahid/000.Damage Severity/aman/embeddings/task_damage_text_img_train.tsv")
df_val = pd.read_table("/home/shahid/000.Damage Severity/aman/embeddings/task_damage_text_img_dev.tsv")
df_test = pd.read_table("/home/shahid/000.Damage Severity/aman/embeddings/task_damage_text_img_test.tsv")
df_train["damage_label"] = df_train["label"].map(label_mapping)
df_val["damage_label"] = df_val["label"].map(label_mapping)
df_test["damage_label"] = df_test["label"].map(label_mapping)

# Convert to arrays
y_train = tf.keras.utils.to_categorical(df_train["damage_label"].values, num_classes=3)
y_val = tf.keras.utils.to_categorical(df_val["damage_label"].values, num_classes=3)
y_test = tf.keras.utils.to_categorical(df_test["damage_label"].values, num_classes=3)

class L1Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, l1=0.01):
        self.l1 = l1

    def __call__(self, x):
        return self.l1 * tf.reduce_sum(tf.abs(x))

    def get_config(self):
        return {'l1': float(self.l1)}

class TripletAttention(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16, attention_dim=8, l1_lambda=0.01):
        super(TripletAttention, self).__init__()
        # Multi-scale convolutions for each attention branch
        self.conv_c1 = tf.keras.layers.Conv2D(1, kernel_size=1, activation='sigmoid')
        self.conv_c3 = tf.keras.layers.SeparableConv2D(1, kernel_size=3, padding='same', activation='sigmoid')
        self.conv_c_dilated = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', dilation_rate=2, activation='sigmoid')
        
        self.conv_h1 = tf.keras.layers.Conv2D(1, kernel_size=1, activation='sigmoid')
        self.conv_h3 = tf.keras.layers.SeparableConv2D(1, kernel_size=3, padding='same', activation='sigmoid')
        self.conv_h_dilated = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', dilation_rate=2, activation='sigmoid')
        
        self.conv_w1 = tf.keras.layers.Conv2D(1, kernel_size=1, activation='sigmoid')
        self.conv_w3 = tf.keras.layers.SeparableConv2D(1, kernel_size=3, padding='same', activation='sigmoid')
        self.conv_w_dilated = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', dilation_rate=2, activation='sigmoid')
        
        # Learnable weights for combining attention branches with L1 regularization
        self.branch_weights = self.add_weight(
            name='branch_weights',
            shape=(3,),
            initializer='uniform',
            regularizer=L1Regularizer(l1=l1_lambda),
            trainable=True
        )
        
        # Global context
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.global_dense = tf.keras.layers.Dense(units=reduction_ratio, activation='relu')
        self.global_dense_restore = tf.keras.layers.Dense(units=1, activation='sigmoid')
        
        # Lightweight self-attention for cross-dimensional interaction
        self.attention_dim = attention_dim
        self.query_conv = tf.keras.layers.Conv2D(attention_dim, kernel_size=1, kernel_regularizer=L1Regularizer(l1=l1_lambda))
        self.key_conv = tf.keras.layers.Conv2D(attention_dim, kernel_size=1, kernel_regularizer=L1Regularizer(l1=l1_lambda))
        self.value_conv = tf.keras.layers.Conv2D(attention_dim, kernel_size=1, kernel_regularizer=L1Regularizer(l1=l1_lambda))
        self.out_conv = tf.keras.layers.Conv2D(1, kernel_size=1, activation='sigmoid')
        
        # Normalization
        self.norm = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])

    def call(self, x):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Global context
        global_context = self.global_pool(x)
        global_context = self.global_dense(global_context)
        global_context = self.global_dense_restore(global_context)
        global_context = tf.tile(global_context, [1, H, W, 1])
        
        # Channel Attention
        x_perm_c = tf.transpose(x, [0, 3, 1, 2])
        ca = tf.reduce_mean(x_perm_c, axis=[2, 3], keepdims=True)
        ca = tf.transpose(ca, [0, 2, 3, 1])
        ca1 = self.conv_c1(ca)
        ca3 = self.conv_c3(ca)
        ca_dilated = self.conv_c_dilated(ca)
        ca = (ca1 + ca3 + ca_dilated) / 3.0
        ca = tf.transpose(ca, [0, 3, 1, 2])
        x_c = tf.transpose(x_perm_c * ca, [0, 2, 3, 1])
        
        # Height Attention
        x_perm_h = tf.transpose(x, [0, 2, 1, 3])
        ha = tf.reduce_mean(x_perm_h, axis=1, keepdims=True)
        ha1 = self.conv_h1(ha)
        ha3 = self.conv_h3(ha)
        ha_dilated = self.conv_h_dilated(ha)
        ha = (ha1 + ha3 + ha_dilated) / 3.0
        ha = tf.transpose(ha, [0, 2, 1, 3])
        x_h = x * ha
        
        # Width Attention
        wa = tf.reduce_mean(x, axis=1, keepdims=True)
        wa1 = self.conv_w1(wa)
        wa3 = self.conv_w3(wa)
        wa_dilated = self.conv_w_dilated(wa)
        wa = (wa1 + wa3 + wa_dilated) / 3.0
        wa = tf.tile(wa, [1, H, 1, C])
        x_w = x * wa
        
        # Learnable weighting of attention branches
        weights = tf.nn.softmax(self.branch_weights)
        combined = weights[0] * x_c + weights[1] * x_h + weights[2] * x_w
        
        # Reshape attention maps to [B, H, W, 1]
        ca_reshaped = tf.reduce_mean(x_c, axis=-1, keepdims=True)
        ha_reshaped = tf.reduce_mean(x_h, axis=-1, keepdims=True)
        wa_reshaped = tf.reduce_mean(x_w, axis=-1, keepdims=True)
        
        # Lightweight self-attention for cross-dimensional interaction
        combined_attention = tf.concat([ca_reshaped, ha_reshaped, wa_reshaped], axis=-1)  # [B, H, W, 3]
        # Reshape to [B, H*W, 3] for self-attention
        combined_attention = tf.reshape(combined_attention, [B, H * W, 3])
        
        # Compute query, key, value
        query = self.query_conv(tf.reshape(combined_attention, [B, H, W, 3]))  # [B, H, W, attention_dim]
        key = self.key_conv(tf.reshape(combined_attention, [B, H, W, 3]))      # [B, H, W, attention_dim]
        value = self.value_conv(tf.reshape(combined_attention, [B, H, W, 3]))  # [B, H, W, attention_dim]
        
        # Reshape for attention computation
        query = tf.reshape(query, [B, H * W, self.attention_dim])
        key = tf.reshape(key, [B, H * W, self.attention_dim])
        value = tf.reshape(value, [B, H * W, self.attention_dim])
        
        # Self-attention: (Q * K^T) / sqrt(d_k)
        attention_scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(self.attention_dim, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_weights, value)  # [B, H*W, attention_dim]
        
        # Reshape back to [B, H, W, attention_dim]
        attention_output = tf.reshape(attention_output, [B, H, W, self.attention_dim])
        cross_attention = self.out_conv(attention_output)  # [B, H, W, 1]
        
        # Combine with global context and apply normalization
        out = combined * cross_attention * global_context
        out = self.norm(out + x)  # Residual connection with normalization
        
        return out

# -------------------------
# Custom Contrastive Loss
# -------------------------
class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def call(self, z1, z2, labels=None):
        z1 = tf.math.l2_normalize(z1, axis=1)
        z2 = tf.math.l2_normalize(z2, axis=1)
        similarity_matrix = tf.matmul(z1, z2, transpose_b=True) / self.temperature
        batch_size = tf.shape(z1)[0]
        labels = tf.range(batch_size, dtype=tf.int32)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels, similarity_matrix)
        )
        return loss


# -------------------------
# Uncertainty Weighted Fusion
# -------------------------
class UncertaintyWeightedFusion(tf.keras.layers.Layer):
    def __init__(self, units=128, mc_runs=3, dropout_rate=0.1, temperature=0.5, use_residual=True):
        super().__init__()
        self.mc_runs = mc_runs
        self.dropout_rate = dropout_rate
        self.temperature = tf.Variable(initial_value=temperature, trainable=True, dtype=tf.float32)
        self.dense = layers.Dense(units, activation='relu')
        self.norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout_rate)
        self.post_proj = layers.Dense(units, activation='relu')
        self.use_residual = use_residual
        self.epsilon = 1e-6

    def call(self, cnn_feat, graph_feat, training=False):
        cnn_proj = self.norm(self.dense(cnn_feat))
        graph_proj = self.norm(self.dense(graph_feat))

        if training:
            # Residual dropout strategy for MC sampling
            cnn_preds = tf.stack([
                cnn_proj + self.dropout(cnn_proj, training=True)
                for _ in range(self.mc_runs)
            ], axis=0)

            graph_preds = tf.stack([
                graph_proj + self.dropout(graph_proj, training=True)
                for _ in range(self.mc_runs)
            ], axis=0)

            cnn_std = tf.math.reduce_std(cnn_preds, axis=0)
            graph_std = tf.math.reduce_std(graph_preds, axis=0)

            cnn_std = tf.clip_by_value(cnn_std, self.epsilon, 1e2)
            graph_std = tf.clip_by_value(graph_std, self.epsilon, 1e2)

            cnn_uncertainty = tf.reduce_mean(cnn_std, axis=1)
            graph_uncertainty = tf.reduce_mean(graph_std, axis=1)

            # Detach uncertainty from graph (optional but improves stability)
            cnn_uncertainty = tf.stop_gradient(cnn_uncertainty)
            graph_uncertainty = tf.stop_gradient(graph_uncertainty)

            logits = tf.stack([-cnn_uncertainty, -graph_uncertainty], axis=1)
            weights = tf.nn.softmax(logits / self.temperature, axis=1)

            weight_cnn = tf.expand_dims(weights[:, 0], axis=-1)
            weight_graph = tf.expand_dims(weights[:, 1], axis=-1)
        else:
            weight_cnn = weight_graph = 0.5

        fused = weight_cnn * cnn_proj + weight_graph * graph_proj

        if self.use_residual:
            fused += 0.5 * (cnn_proj + graph_proj)  # Residual skip

        fused = self.post_proj(fused)
        tf.debugging.check_numerics(fused, "Fused output has NaNs")
        return fused


# -------------------------
# Cross-modal Graph Refinement
# -------------------------
class CrossModalGraphRefinement(tf.keras.layers.Layer):
    def __init__(self, units, num_heads=4):
        super().__init__()
        self.units = units
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=units // num_heads)
        self.gate = layers.Dense(units, activation='sigmoid')
        self.norm = layers.LayerNormalization()
        self.img_dense = layers.Dense(units)

    def call(self, graph_emb, img_feat):
        img_proj = self.img_dense(img_feat)
        graph_emb_expanded = tf.expand_dims(graph_emb, axis=1)
        img_proj_expanded = tf.expand_dims(img_proj, axis=1)
        refined_graph = self.attention(query=img_proj_expanded, value=graph_emb_expanded, key=graph_emb_expanded)
        refined_graph = tf.squeeze(refined_graph, axis=1)
        gate = self.gate(img_proj_expanded)
        gate = tf.squeeze(gate, axis=1)
        refined_graph = gate * refined_graph + (1 - gate) * graph_emb
        refined_graph = self.norm(refined_graph)
        return refined_graph

# -------------------------
# Build the Model
# -------------------------
def build_model(num_classes=3, input_shape_cnn=(49, 1024), input_shape_graph=(128,), handcraft_input_shape=(13,), exp_input_shape=(9,)):
    raw_input = tf.keras.Input(shape=input_shape_cnn, name="raw_swin_input")
    graph_input = tf.keras.Input(shape=input_shape_graph, name="graphsage_input")
    handcraft_input = tf.keras.Input(shape=handcraft_input_shape, name="handcraft_input")
    exp_input = tf.keras.Input(shape=exp_input_shape, name="exp_input")

    # CNN Branch
    reshaped = layers.Reshape((7, 7, 1024))(raw_input)
    x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')(reshaped)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = TripletAttention()(x)
    cnn_feat = layers.GlobalAveragePooling2D()(x)
    cnn_proj = layers.Dense(128, activation='relu')(cnn_feat)

    # Graph Branch
    graph_proj = layers.Dense(128, activation='relu')(graph_input)
    graph_proj = layers.Dropout(0.2)(graph_proj)

    # Cross-modal Graph Refinement
    refined_graph = CrossModalGraphRefinement(units=128)(graph_proj, cnn_proj)

    # Uncertainty-guided Fusion
    fused = UncertaintyWeightedFusion(units=128)(cnn_proj, refined_graph)
    
    #for handcrafted features
    h = layers.LayerNormalization()(handcraft_input)
    h=layers.Dense(13, activation='relu')(h)
    # ex = layers.LayerNormalization()(exp_input)
    ex=layers.Dense(9, activation='relu')(exp_input)
    con = layers.Concatenate()([ h, ex])
    con = layers.Dense(22, activation='relu')(con)

    
    print(ex.shape)
    # Classifier
    y = layers.Dense(128, activation='relu')(fused)
    y = layers.Dropout(0.2)(y)
    y = layers.Concatenate()([y, con])
    y = layers.Dense(128, activation='relu')(y)
    y = layers.Dropout(0.2)(y)
    output = layers.Dense(num_classes, activation='softmax', name="classification_output")(y)

    # Normalized embeddings for contrastive loss
    cnn_norm = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="cnn_norm")(cnn_proj)
    graph_norm = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="graph_norm")(fused)

    model = Model(inputs=[graph_input, raw_input,handcraft_input, exp_input], outputs=[output, cnn_norm, graph_norm])
    return model

# -------------------------
# Instantiate Model & Losses
# -------------------------
model = build_model(num_classes=3)
classification_loss = tf.keras.losses.CategoricalCrossentropy()
contrastive_loss = ContrastiveLoss(temperature=0.5)
# optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-3)

# -------------------------
# Custom Training Step
# -------------------------
@tf.function
def train_step(inputs, labels):
    graph_input, raw_input,handcraft_input, exp_input = inputs
    with tf.GradientTape() as tape:
        class_output, cnn_norm_val, graph_norm_val = model([graph_input, raw_input, handcraft_input, exp_input], training=True)
        cls_loss = classification_loss(labels, class_output)
        cont_loss = contrastive_loss(cnn_norm_val, graph_norm_val)
        total_loss = cls_loss + 0.5 * cont_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss, cls_loss, cont_loss


# Update optimizer
optimizer = AdamW(learning_rate=1e-2, weight_decay=1e-2)

# -------------------------
# Compile Model for Evaluation
# -------------------------
model.compile(
    optimizer=optimizer,
    loss={"classification_output": classification_loss},
    metrics={"classification_output": ['accuracy', Precision(), Recall()]})


model.summary()

# === Early Stopping Configuration ===
best_val_loss = np.inf
patience = 5
wait = 0
best_weights = None
early_stop_triggered = False

# === Training Loop ===
epochs = 100
batch_size = 32
steps_per_epoch = len(graph_img_train) // batch_size

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    
    # Shuffle training data (optional)
    indices = np.arange(len(graph_img_train))
    np.random.shuffle(indices)
    
    graph_img_train = graph_img_train[indices]
    train_image_raw = train_image_raw[indices]
    handcraft_img_train = handcraft_img_train[indices]
    exp_img_train = exp_img_train[indices]
    y_train = y_train[indices]

    # Train loop
    for step in range(steps_per_epoch):
        batch_inputs = (
            graph_img_train[step*batch_size:(step+1)*batch_size],
            train_image_raw[step*batch_size:(step+1)*batch_size],
            handcraft_img_train[step*batch_size:(step+1)*batch_size],
            exp_img_train[step*batch_size:(step+1)*batch_size]
        )
        batch_labels = y_train[step*batch_size:(step+1)*batch_size]
        
        total_loss, cls_loss, cont_loss = train_step(batch_inputs, batch_labels)
        print(f"Step {step+1}/{steps_per_epoch}: Loss = {total_loss:.4f} (Cls = {cls_loss:.4f}, Cont = {cont_loss:.4f})")
    
    # === Validation Evaluation (on full val set) ===
    val_preds = model([graph_img_val, val_image_raw, handcraft_img_val, exp_img_val], training=False)[0]
    val_loss = tf.keras.losses.categorical_crossentropy(y_val, val_preds)
    val_loss = tf.reduce_mean(val_loss).numpy()
    
    print(f" Validation Loss = {val_loss:.4f}")
    
    # === Early Stopping Logic ===
    if val_loss < best_val_loss:
        print("Validation loss improved.")
        best_val_loss = val_loss
        best_weights = model.get_weights()
        wait = 0
    else:
        wait += 1
        print(f" No improvement. Wait count = {wait}/{patience}")
        if wait >= patience:
            print("Early stopping triggered.")
            early_stop_triggered = True
            break

# === Restore Best Weights ===
if early_stop_triggered and best_weights is not None:
    print("Restoring best model weights.")
    model.set_weights(best_weights)


eval_model = Model(
    inputs=model.inputs,
    outputs=model.get_layer("classification_output").output
)
eval_model.compile(
    optimizer=optimizer,
    loss=classification_loss,
    metrics=['accuracy', Precision(), Recall()]
)

test_results = eval_model.evaluate(
    [graph_img_test, test_image_raw,handcraft_img_test,exp_img_test],
    y_test,
    batch_size=32,
    verbose=1
)
print("\n Final Test Results:")
print(f"Loss: {test_results[0]:.16f}")
print(f"Accuracy: {test_results[1]:.16f}")
print(f"Precision: {test_results[2]:.16f}")
print(f"Recall: {test_results[3]:.16f}")


# Make predictions
predictions = model.predict([graph_img_test, test_image_raw,handcraft_img_test,exp_img_test], batch_size=32)

# Extract classification output (first output)
class_predictions = predictions[0]  # Shape: (num_samples, 3)

predicted_labels = np.argmax(class_predictions, axis=1)  # Shape: (num_samples,)
y_test_labels = np.argmax(y_test, axis=1)

# Step 3: Sanity check to ensure lengths match
assert len(predicted_labels) == len(df_test), "Mismatch between predicted labels and test file rows!"

# Step 4: Add the predicted labels as a new column
df_test["Proposed_method"] = predicted_labels

# Step 5: Save the updated DataFrame
df_test.to_csv("test_with_predictions.csv", index=False)

print("Done! The file has been saved as 'test_with_predictions.csv'")


y_prediction=predicted_labels
test_labels=y_test_labels
# Compute accuracy
accuracy = accuracy_score(test_labels, y_prediction)

# Compute Precision
precision_micro = precision_score(test_labels, y_prediction, average='micro')
precision_macro = precision_score(test_labels, y_prediction, average='macro')
precision_weighted = precision_score(test_labels, y_prediction, average='weighted')

# Compute Recall
recall_micro = recall_score(test_labels, y_prediction, average='micro')
recall_macro = recall_score(test_labels, y_prediction, average='macro')
recall_weighted = recall_score(test_labels, y_prediction, average='weighted')

# Compute F1-Score
f1_micro = f1_score(test_labels, y_prediction, average='micro')
f1_macro = f1_score(test_labels, y_prediction, average='macro')
f1_weighted = f1_score(test_labels, y_prediction, average='weighted')

# Store all metrics in a dictionary
metrics = {
    'accuracy': accuracy,

    'precision_micro': precision_micro,
    'precision_macro': precision_macro,
    'precision_weighted': precision_weighted,

    'recall_micro': recall_micro,
    'recall_macro': recall_macro,
    'recall_weighted': recall_weighted,

    'f1_micro': f1_micro,
    'f1_macro': f1_macro,
    'f1_weighted': f1_weighted
}

# Print all metrics nicely
for key, value in metrics.items():
    print(f"{key}: {value:.16f}")

model.save("model.h5")



# Save F1-scores to a file
file_path = "model_with_novel_hand_exp_features1.txt"
with open(file_path, 'w') as file:
    for score_name, score_value in metrics.items():
        file.write(f"{score_name}: {score_value:.16f}\n")


from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_curve, auc,
                             precision_recall_curve)
from sklearn.preprocessing import label_binarize

# Define the save directory
save_dir = "results2_final"
os.makedirs(save_dir, exist_ok=True)

# Generate report and confusion matrix
report = classification_report(test_labels, y_prediction, output_dict=True)
per_class_df = pd.DataFrame(report).transpose()

cm = confusion_matrix(test_labels, y_prediction)
n_classes = len(np.unique(test_labels))
y_true_bin = label_binarize(test_labels, classes=np.arange(n_classes))
y_pred_prob = y_prediction if y_prediction.ndim > 1 else label_binarize(y_prediction, classes=np.arange(n_classes))

# Set up 1 row, 3 column subplot
fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)

# -------------------------------
# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False,
            annot_kws={"size": 10})
axes[0].set_title("Confusion Matrix", fontsize=14)
axes[0].set_xlabel("Predicted Label", fontsize=12)
axes[0].set_ylabel("True Label", fontsize=12)
axes[0].tick_params(axis='both', labelsize=10)

# -------------------------------
# 2. Precision-Recall Curve
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_prob[:, i])
    axes[1].plot(recall, precision, label=f"Class {i}")
axes[1].set_title("Precision-Recall Curves", fontsize=14)
axes[1].set_xlabel("Recall", fontsize=12)
axes[1].set_ylabel("Precision", fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True)
axes[1].tick_params(axis='both', labelsize=10)

# -------------------------------
# 3. ROC Curve
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    axes[2].plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
axes[2].plot([0, 1], [0, 1], "k--")
axes[2].set_title("ROC Curves", fontsize=14)
axes[2].set_xlabel("False Positive Rate", fontsize=12)
axes[2].set_ylabel("True Positive Rate", fontsize=12)
axes[2].legend(fontsize=10)
axes[2].grid(True)
axes[2].tick_params(axis='both', labelsize=10)

# Save the combined figure
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "combined_metrics_grid.png"))
plt.close()

# -------------------------------
# Save metrics
metrics = {
    "accuracy": accuracy_score(test_labels, y_prediction),
    "precision_macro": precision_score(test_labels, y_prediction, average='macro'),
    "recall_macro": recall_score(test_labels, y_prediction, average='macro'),
    "f1_macro": f1_score(test_labels, y_prediction, average='macro')
}

with open(os.path.join(save_dir, "local_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

per_class_df.to_csv(os.path.join(save_dir, "per_class_metrics.csv"))

# -------------------------------

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import json

# from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,classification_report, confusion_matrix, roc_curve, auc,precision_recall_curve)
# from sklearn.preprocessing import label_binarize

# import os

# # Define the save directory
# save_dir = "results2"  # Change to your desired output path
# os.makedirs(save_dir, exist_ok=True)

# # Per-class metrics
# report = classification_report(test_labels, y_prediction, output_dict=True)
# per_class_df = pd.DataFrame(report).transpose()

# # Confusion Matrix
# cm = confusion_matrix(test_labels, y_prediction)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
# plt.close()
# plt.show()

# # PR and ROC Curves
# n_classes = len(np.unique(test_labels))
# y_true_bin = label_binarize(test_labels, classes=np.arange(n_classes))
# y_pred_prob = y_prediction if y_prediction.ndim > 1 else label_binarize(y_prediction, classes=np.arange(n_classes))

# # Precision-Recall Curve
# plt.figure(figsize=(10, 7))
# for i in range(n_classes):
#     precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_prob[:, i])
#     plt.plot(recall, precision, label=f"Class {i}")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curves")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(save_dir, "precision_recall_curve.png"))
# plt.close()
# plt.show()

# # ROC Curve
# plt.figure(figsize=(10, 7))
# for i in range(n_classes):
#     fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
# plt.plot([0, 1], [0, 1], "k--")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curves")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(save_dir, "roc_curve.png"))
# plt.close()
# plt.show()

# # Save metrics and report
# with open(os.path.join(save_dir, "local_metrics.json"), "w") as f:
#     json.dump(metrics, f, indent=4)

# per_class_df.to_csv(os.path.join(save_dir, "per_class_metrics.csv"))


