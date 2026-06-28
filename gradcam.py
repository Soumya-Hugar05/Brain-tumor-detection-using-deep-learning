from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf


def _find_feature_model(model: tf.keras.Model) -> tf.keras.Model:
    feature_model = next(
        (layer for layer in model.layers if isinstance(layer, tf.keras.Model)),
        None,
    )
    if feature_model is None:
        raise RuntimeError("Could not find nested feature extractor model for Grad-CAM.")
    return feature_model


def make_gradcam_heatmap(
    model: tf.keras.Model,
    preprocessed_input: np.ndarray,
    class_index: int | None = None,
    last_conv_layer_name: str | None = None,
) -> np.ndarray:
    feature_model = _find_feature_model(model)
    classifier_input = tf.keras.Input(shape=feature_model.output.shape[1:])
    x_head = classifier_input

    passed_feature_model = False
    for layer in model.layers:
        if layer is feature_model:
            passed_feature_model = True
            continue
        if not passed_feature_model or isinstance(layer, tf.keras.layers.InputLayer):
            continue
        x_head = layer(x_head)

    classifier_model = tf.keras.Model(classifier_input, x_head)

    x = tf.convert_to_tensor(preprocessed_input)
    if x.ndim == 3:
        x = tf.expand_dims(x, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs = feature_model(x, training=False)
        tape.watch(conv_outputs)
        preds = classifier_model(conv_outputs)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(
    rgb_image_uint8: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    h, w = rgb_image_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = np.clip((1 - alpha) * rgb_image_uint8 + alpha * heatmap_color, 0, 255).astype(
        np.uint8
    )
    return overlay

