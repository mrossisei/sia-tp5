import numpy as np


# ---------------------------------------------------------------------------
# Métricas específicas del TP5 (autoencoders sobre datos binarios)
# ---------------------------------------------------------------------------

def pixel_errors(X_true, X_pred, threshold=0.5):
    """Cantidad de píxeles incorrectos por patrón.

    X_true: (N, D) en {0,1}. X_pred: (N, D) en (0,1) (salida sigmoide).
    Devuelve un array (N,) de enteros: píxeles mal reconstruidos por muestra.
    """
    X_true = np.asarray(X_true)
    X_pred = np.asarray(X_pred)
    pred_bin = (X_pred >= threshold).astype(int)
    return np.sum(pred_bin != X_true.astype(int), axis=1)


def pixel_error_summary(X_true, X_pred, threshold=0.5):
    """Resumen del criterio de éxito del EJ1.a ('por patrón: máx ≤ 1').

    Devuelve un dict con: per_pattern (array N,), max, mean, total,
    n_exact (0 errores), n_le1 (≤1 error) y success (bool: max ≤ 1).
    """
    errs = pixel_errors(X_true, X_pred, threshold=threshold)
    return {
        "per_pattern": errs,
        "max": int(errs.max()) if errs.size else 0,
        "mean": float(errs.mean()) if errs.size else 0.0,
        "total": int(errs.sum()),
        "n_exact": int(np.sum(errs == 0)),
        "n_le1": int(np.sum(errs <= 1)),
        "n": int(errs.size),
        "success": bool(errs.size and errs.max() <= 1),
    }


# ---------------------------------------------------------------------------
# Métricas de clasificación portadas de TP3 (disponibles por consistencia)
# ---------------------------------------------------------------------------

def confusion_matrix(y_true, y_pred):
    """Returns [[TN, FP], [FN, TP]]."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return np.array([[tn, fp], [fn, tp]])


def confusion_matrix_multiclass(y_true, y_pred, n_classes=None):
    """Return n_classes x n_classes confusion matrix.

    cm[i, j] = number of samples with true class i predicted as class j.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return float(precision), float(recall), float(f1)
