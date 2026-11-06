"""Optional expansion of a bounded ranking with additional session evidence."""
from __future__ import annotations

import numpy as np


def dense_session_tail(ranking, dense, sessions):
    """Protect twenty results and add five representatives of uncovered sessions."""
    order = list(ranking)
    scores = np.asarray(dense, dtype=np.float64)
    count = len(sessions)
    if (scores.shape != (count,) or not np.isfinite(scores).all()
            or len(order) != count or set(order) != set(range(count))):
        raise ValueError('invalid session expansion ranking or scores')
    # Empty session identifiers carry no evidence of a shared source session.
    def session(index):
        value = sessions[index]
        return value if isinstance(value, str) and value.strip() else None

    limit = min(100, count)
    covered = {session(i) for i in order[:limit]}
    additions = []
    for index in sorted(range(count), key=lambda i: (-scores[i], i)):
        key = session(index)
        if key is not None and key not in covered:
            additions.append(index)
            covered.add(key)
            if len(additions) == 5:
                break
    head = order[:min(20, count)]
    return list(dict.fromkeys(head + additions + order[20:]))


def dense_session_image_tail(ranking, dense, sessions, image_mask):
    """Add image representatives for head sessions lacking an image in the hundred."""
    order = list(ranking)
    scores = np.asarray(dense, dtype=np.float64)
    count = len(sessions)
    if (scores.shape != (count,) or not np.isfinite(scores).all()
            or len(order) != count or set(order) != set(range(count))
            or len(image_mask) != count
            or any(not isinstance(value, (bool, np.bool_)) for value in image_mask)):
        raise ValueError('invalid session image support inputs')
    def session(index):
        value = sessions[index]
        return value if isinstance(value, str) and value.strip() else None

    head = order[:min(20, count)]
    eligible = {session(i) for i in head} - {None}
    covered = {session(i) for i in order[:100] if image_mask[i]}
    added = []
    for index in sorted(range(count), key=lambda i: (-scores[i], i)):
        key = session(index)
        if image_mask[index] and key in eligible and key not in covered:
            added.append(index)
            covered.add(key)
            if len(added) == 5:
                break
    return list(dict.fromkeys(head + added + order[20:]))
