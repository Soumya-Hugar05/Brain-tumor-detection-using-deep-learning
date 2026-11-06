"""SVM-faithful Grad-CAM.

The final classifier is  image -> ResNet50 conv5 map A (7x7x2048) -> GAP -> f (2048)
                               -> StandardScaler -> SVC (one-vs-one, RBF or linear)
so the SVM's class score is a differentiable function of A. This module re-implements the
fitted scaler + SVC decision function in TensorFlow (from support vectors, dual coefficients,
intercepts and gamma), verifies it against scikit-learn, and computes Grad-CAM with the
gradient of the SVM's own class score - no surrogate CNN head is involved.

Class score explained (= sklearn SVC.decision_function, decision_function_shape='ovr'):
    s_c = votes_c + S_c / (3 (|S_c| + 1)),  S_c = sum of signed one-vs-one decision values for c
votes_c is piecewise constant (zero gradient), so the gradient comes from S_c through a
monotone transform. With flip-averaged features f = (GAP(A(x)) + GAP(A(flip x))) / 2 the map is
cam = sum_k a_k A_k(x) + unflip(sum_k a'_k A_k(flip x)).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input


class TFSVM:
    """TensorFlow copy of a fitted Pipeline(StandardScaler, SVC)."""

    def __init__(self, pipe, dtype=tf.float64):
        scaler, svc = pipe["scaler"], pipe["svm"]
        if svc.kernel not in ("rbf", "linear"):
            raise ValueError(f"unsupported kernel {svc.kernel}")
        self.dtype = dtype
        self.k = len(svc.classes_)
        assert list(svc.classes_) == list(range(self.k))
        c = lambda a: tf.constant(np.asarray(a, dtype=np.float64), dtype=dtype)
        self.mean, self.scale = c(scaler.mean_), c(scaler.scale_)
        self.sv, self.dual, self.intercept = c(svc.support_vectors_), c(svc.dual_coef_), c(svc.intercept_)
        self.kernel, self.gamma = svc.kernel, float(svc._gamma)
        self.sv_sq = tf.reduce_sum(self.sv ** 2, axis=1)
        starts = np.concatenate([[0], np.cumsum(svc.n_support_)])
        self.slices = [(int(starts[i]), int(starts[i + 1])) for i in range(self.k)]

    def kernel_matrix(self, z):
        dot = tf.matmul(z, self.sv, transpose_b=True)
        if self.kernel == "linear":
            return dot
        d2 = tf.reduce_sum(z ** 2, axis=1, keepdims=True) - 2.0 * dot + self.sv_sq[None, :]
        return tf.exp(-self.gamma * tf.maximum(d2, 0.0))

    def ovo(self, feats):
        """One-vs-one decision values (N, k(k-1)/2); positive favours the lower class index."""
        z = (tf.cast(feats, self.dtype) - self.mean) / self.scale
        K = self.kernel_matrix(z)
        out, p = [], 0
        for i in range(self.k):
            for j in range(i + 1, self.k):
                (a0, a1), (b0, b1) = self.slices[i], self.slices[j]
                d = (tf.linalg.matvec(K[:, a0:a1], self.dual[j - 1, a0:a1])
                     + tf.linalg.matvec(K[:, b0:b1], self.dual[i, b0:b1]) + self.intercept[p])
                out.append(d)
                p += 1
        return tf.stack(out, axis=1)

    def conf_sums(self, feats):
        """S_c: sum of one-vs-one decision values signed towards class c (continuous margin)."""
        dec = self.ovo(feats)
        cols, p = [[] for _ in range(self.k)], 0
        for i in range(self.k):
            for j in range(i + 1, self.k):
                cols[i].append(dec[:, p])
                cols[j].append(-dec[:, p])
                p += 1
        return tf.stack([tf.add_n(c) for c in cols], axis=1), dec

    def ovr(self, feats):
        """Exactly sklearn's SVC.decision_function (ovr shape)."""
        S, dec = self.conf_sums(feats)
        votes, p = [tf.zeros_like(S[:, 0]) for _ in range(self.k)], 0
        for i in range(self.k):
            for j in range(i + 1, self.k):
                win_i = tf.cast(dec[:, p] > 0, self.dtype)
                votes[i] = votes[i] + win_i
                votes[j] = votes[j] + (1.0 - win_i)
                p += 1
        return tf.stack(votes, axis=1) + S / (3.0 * (tf.abs(S) + 1.0))


class SVMGradCAM:
    """Grad-CAM whose target is the SVM decision score (float32 graph)."""

    def __init__(self, backbone: tf.keras.Model, pipe, flip_average: bool):
        self.backbone = backbone
        self.svm = TFSVM(pipe, dtype=tf.float32)
        self.flip_average = flip_average

        @tf.function(reduce_retracing=True)
        def _cam(x, cls):
            with tf.GradientTape() as tape:
                A = self.backbone(preprocess_input(x), training=False)
                maps = [A]
                if self.flip_average:
                    A2 = self.backbone(preprocess_input(tf.image.flip_left_right(x)), training=False)
                    maps.append(A2)
                    f = (tf.reduce_mean(A, [1, 2]) + tf.reduce_mean(A2, [1, 2])) / 2.0
                else:
                    f = tf.reduce_mean(A, [1, 2])
                scores = self.svm.ovr(f)
                target = tf.reduce_sum(tf.gather(scores, cls, axis=1, batch_dims=1))
            grads = tape.gradient(target, maps)
            cam = tf.reduce_sum(tf.reduce_mean(grads[0], [1, 2], keepdims=True) * maps[0], axis=-1)
            if self.flip_average:
                cam2 = tf.reduce_sum(tf.reduce_mean(grads[1], [1, 2], keepdims=True) * maps[1], axis=-1)
                cam = cam + tf.reverse(cam2, axis=[2])
            return cam, scores

        self._cam = _cam

    def __call__(self, images: np.ndarray, class_index: np.ndarray | None = None):
        """images (N,224,224,3) float [0,255]. Returns heatmaps (N,7,7) in [0,1], svm scores, classes."""
        x = tf.convert_to_tensor(images, tf.float32)
        if class_index is None:
            f = tf.reduce_mean(self.backbone(preprocess_input(x), training=False), [1, 2])
            if self.flip_average:
                f2 = tf.reduce_mean(self.backbone(preprocess_input(tf.image.flip_left_right(x)), training=False), [1, 2])
                f = (f + f2) / 2.0
            class_index = tf.argmax(self.svm.ovr(f), axis=1).numpy()
        cam, scores = self._cam(x, tf.constant(np.asarray(class_index), tf.int32))
        cam = tf.nn.relu(cam).numpy()
        cam = cam / (cam.reshape(len(cam), -1).max(1)[:, None, None] + 1e-8)
        return cam, scores.numpy(), np.asarray(class_index)


def legacy_cnn_gradcam(cnn_model, images: np.ndarray, class_index: np.ndarray) -> np.ndarray:
    """The original project's Grad-CAM (gradcam.py, unmodified) on the CNN head."""
    from gradcam import make_gradcam_heatmap

    out = []
    for img, c in zip(images, class_index):
        out.append(make_gradcam_heatmap(cnn_model, preprocess_input(img.copy()), class_index=int(c)))
    return np.stack(out)


def upsample(heat: np.ndarray, size: int = 224) -> np.ndarray:
    return tf.image.resize(heat[..., None], (size, size), method="bilinear").numpy()[..., 0]
