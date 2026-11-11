"""시행 단위 평가와 데이터 분할 검증. 중첩 윈도우를 독립 시행으로 세지 않는다."""

import numpy as np


def check_splits(splits):
    seen_trials, seen_sessions = set(), set()
    for name, rows in splits.items():
        trials = [row["trial_id"] for row in rows]
        sessions = {row["session_id"] for row in rows}
        if len(set(trials)) != len(trials) or seen_trials.intersection(trials) or seen_sessions.intersection(sessions):
            raise ValueError("시행 또는 세션이 데이터 분할 사이에 중복되었습니다.")
        if not rows or name not in {"training", "validation", "final_test"}:
            raise ValueError("데이터 분할이 비어 있거나 이름이 잘못되었습니다.")
        seen_trials.update(trials)
        seen_sessions.update(sessions)


def trial_metrics(labels, predictions, class_count):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if (class_count < 2 or labels.ndim != 1 or predictions.shape != labels.shape or len(labels) == 0
            or not np.issubdtype(labels.dtype, np.integer) or not np.issubdtype(predictions.dtype, np.integer)
            or np.any((labels < 0) | (labels >= class_count))
            or np.any((predictions < -1) | (predictions >= class_count))):
        raise ValueError("시행별 라벨·예측·클래스 수가 잘못되었습니다.")
    # 마지막 열은 거부(-1)이며 실제 REST 클래스와 합치지 않는다.
    confusion = np.zeros((class_count, class_count + 1), dtype=int)
    for label, predicted in zip(labels, predictions):
        confusion[label, class_count if predicted == -1 else predicted] += 1
    totals = confusion.sum(axis=1)
    if np.any(totals == 0):
        raise ValueError("평가에 누락된 클래스가 있습니다.")
    recall = np.diag(confusion[:, :class_count]) / totals
    accepted = predictions != -1
    correct = predictions == labels
    return {"trial_count": len(labels), "balanced_accuracy": float(recall.mean()),
            "recall": recall.tolist(), "confusion_with_reject_column": confusion.tolist(),
            "coverage": float(accepted.mean()), "all_trial_accuracy": float(correct.mean()),
            "conditional_accuracy": float(correct[accepted].mean()) if accepted.any() else None}
