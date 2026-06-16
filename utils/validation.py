"""Shared validation and metrics utilities."""


def compute_classification_metrics(df, pred_col='is_anomaly', actual_col='actual_anomaly'):
    """Compute TP, FP, FN, precision, and recall from prediction and ground-truth columns.

    Returns a dict with keys: tp, fp, fn, total_predicted, total_actual, precision, recall.
    """
    total_actual = int(df[actual_col].sum())
    total_predicted = int(df[pred_col].sum())

    tp = int(((df[pred_col] == 1) & (df[actual_col] == 1)).sum())
    fp = int(((df[pred_col] == 1) & (df[actual_col] == 0)).sum())
    fn = int(((df[pred_col] == 0) & (df[actual_col] == 1)).sum())

    precision = tp / total_predicted if total_predicted > 0 else 0.0
    recall = tp / total_actual if total_actual > 0 else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'total_predicted': total_predicted,
        'total_actual': total_actual,
        'precision': precision,
        'recall': recall,
    }


def print_validation_report(metrics):
    """Print a formatted validation report from a metrics dict."""
    print("\n================ VALIDATION REPORT ================")
    print(f"Total Anomalies Injected (Ground Truth): {metrics['total_actual']}")
    print(f"Total Anomalies Flagged by Model      : {metrics['total_predicted']}")
    print("---------------------------------------------------")
    print(f"True Positives (Successfully Caught)  : {metrics['tp']}")
    print(f"False Positives (False Alarms)        : {metrics['fp']}")
    print(f"False Negatives (Missed Anomalies)    : {metrics['fn']}")
    print("---------------------------------------------------")
    print(f"Precision (When it flags, how right is it?): {metrics['precision']:.2%}")
    print(f"Recall (What % of anomalies did it catch?): {metrics['recall']:.2%}")
    print("===================================================\n")
