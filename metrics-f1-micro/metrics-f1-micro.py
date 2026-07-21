def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    if len(y_true) == 0:
        return 0.0

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)