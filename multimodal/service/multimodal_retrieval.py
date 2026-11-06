"""Per-part retrieval with optional calibration of text and image channels."""
from __future__ import annotations

import numpy as np


def rerank_prefix(candidate_ids, relevance, mode='consensus', image_mask=None):
    """Reorder a bounded prefix; preserve all remaining IDs."""
    ids = list(candidate_ids)
    values = np.asarray(relevance, dtype=np.float64)
    count = min(100 if mode == 'protected20_tail100' else (40 if mode == 'protected40' else 20), len(ids))
    if len(set(ids)) != len(ids) or values.shape != (count,):
        raise ValueError('reranker candidate/score mismatch')
    if mode not in {'direct', 'consensus', 'image_locked', 'protected40', 'protected20_tail100'} or not np.isfinite(values).all():
        raise ValueError('invalid reranker mode or scores')
    if mode == 'protected20_tail100':
        prefix_count = min(40, count)
        prefix = rerank_prefix(ids[:prefix_count], values[:prefix_count], 'protected40')
        head = prefix[:20]
        protected = set(head)
        tail = sorted((i for i in range(count) if ids[i] not in protected),
                      key=lambda i: (-values[i], i))
        return head + [ids[i] for i in tail] + ids[count:]
    if mode == 'protected40':
        retained = sorted(range(min(20, count)), key=lambda i: (-values[i], i))
        head = retained[:10]
        protected = set(head)
        tail = sorted((i for i in range(count) if i not in protected),
                      key=lambda i: (-values[i], i))
        return [ids[i] for i in head + tail] + ids[count:]
    if mode == 'image_locked':
        if image_mask is None or len(image_mask) != count or any(type(v) is not bool for v in image_mask):
            raise ValueError('image-locked reranking requires a boolean candidate mask')
        slots = [i for i in range(count) if not image_mask[i]]
        ordered = sorted(slots, key=lambda i: (-values[i], i))
        result = list(ids)
        for slot, index in zip(slots, ordered):
            result[slot] = ids[index]
        return result
    order = np.argsort(-values, kind='stable')
    if mode == 'consensus':
        local = np.empty(count, dtype=np.int64)
        local[order] = np.arange(count)
        order = np.argsort(np.arange(count) + local, kind='stable')
    return [ids[int(i)] for i in order] + ids[count:]


def session_diverse_order(candidate_ids, relevance, session_ids, *, protected_head=0):
    """Rank scorable text slots in the first hundred, preserving protected slots."""
    if type(protected_head) is not int or not 0 <= protected_head <= 100:
        raise ValueError('invalid protected session head')
    ids = list(candidate_ids)
    count = min(100, len(ids))
    if len(set(ids)) != len(ids) or len(relevance) != count or len(session_ids) != count:
        raise ValueError('session reranker candidate/score mismatch')
    if any(s is not None and not isinstance(s, str) for s in session_ids):
        raise ValueError('invalid source session')
    scores = [None if v is None else float(v) for v in relevance]
    if any(v is not None and not np.isfinite(v) for v in scores):
        raise ValueError('nonfinite session relevance')
    def session(i):
        value = session_ids[i]
        return ('session', value) if value and value.strip() else ('record', ids[i])
    available = {i for i, value in enumerate(scores) if value is not None}
    result = list(ids)
    covered = {session(i) for i in range(min(protected_head, count)) if scores[i] is None}
    for dest in range(count):
        if scores[dest] is None:
            covered.add(session(dest))
            continue
        fresh = {i for i in available if session(i) not in covered}
        source = min(fresh or available, key=lambda i: (-scores[i], i))
        result[dest] = ids[source]
        available.remove(source)
        covered.add(session(source))
    return result


def _calibrate(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    result = np.full(values.shape, -np.inf, dtype=np.float64)
    if valid.any():
        std = float(values[valid].std())
        result[valid] = (values[valid] - values[valid].mean()) / std if std > 1e-8 else 0
    return result


def _rrf(values: np.ndarray) -> np.ndarray:
    valid = np.flatnonzero(np.isfinite(values))
    result = np.zeros(values.shape, dtype=np.float64)
    ranking = valid[np.argsort(-values[valid], kind='stable')]
    result[ranking] = 1 / (60 + np.arange(1, len(ranking) + 1))
    return result


def score_vectors(querymatrix, docmatrices, docparttypes, queryparttypes,
                  mode: str = 'late', visual_weight: float = 1) -> np.ndarray:
    """Score normalized vectors; each row corresponds to one ordered content part.

    Image-containing queries retain late interaction. Text-only queries may
    calibrate within-modality scores over this user's available documents.
    Missing channels never enter calibration or receive reciprocal rank votes.
    """
    if mode not in {'late', 'modality_zmax', 'modality_rrf'}:
        raise ValueError('unknown multimodal scoring mode')
    if not np.isfinite(visual_weight) or visual_weight < 0:
        raise ValueError('visual_weight must be finite and nonnegative')
    query = np.asarray(querymatrix)
    if query.ndim != 2 or not len(query) or len(queryparttypes) != len(query):
        raise ValueError('query part/vector shape mismatch')
    if len(docmatrices) != len(docparttypes):
        raise ValueError('document part/vector shape mismatch')
    direct_late = mode == 'late' or 'image_url' in queryparttypes
    late = []
    text = np.full(len(docmatrices), -np.inf)
    image = np.full(len(docmatrices), -np.inf)
    for index, (matrix, types) in enumerate(zip(docmatrices, docparttypes)):
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or len(matrix) != len(types):
            raise ValueError('document part/vector shape mismatch')
        if not len(matrix):
            if matrix.shape[1] not in {0, query.shape[1]} or types:
                raise ValueError('empty document vector shape mismatch')
            late.append(-np.inf)
            continue
        if matrix.shape[1] != query.shape[1]:
            raise ValueError('document part/vector shape mismatch')
        if not np.isfinite(matrix).all() or not np.isfinite(query).all():
            raise ValueError('nonfinite embedding')
        pairwise = query @ matrix.T
        late.append(float(pairwise.max(axis=1).mean()))
        if direct_late:
            continue
        for channel, kind in [(text, 'text'), (image, 'image_url')]:
            selected = [i for i, value in enumerate(types) if value == kind]
            if selected:
                channel[index] = float(pairwise[:, selected].max(axis=1).mean())
    late = np.asarray(late, dtype=np.float64)
    if direct_late or not np.isfinite(text).any() or not np.isfinite(image).any():
        return late
    if mode == 'modality_rrf':
        return _rrf(text) + visual_weight * _rrf(image)
    calibrated_text, calibrated_image = _calibrate(text), _calibrate(image)
    if visual_weight == 0:
        # Documents without text sort behind documents carrying text evidence.
        return calibrated_text
    return np.maximum(calibrated_text, visual_weight * calibrated_image)


def propagate_context(scores, user_ids, session_ids, sequences, weight: float = 0,
                      valid_vectors=None) -> np.ndarray:
    """Blend only immediate storage neighbors within a nonempty user session.

    All updates read the original scores, so propagation is exactly one hop.
    Global sequence gaps and missing vectors form boundaries.
    """
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('context weight must be in [0, 1]')
    original = np.asarray(scores, dtype=np.float64)
    if original.ndim != 1 or any(len(values) != len(original) for values in (user_ids, session_ids, sequences)):
        raise ValueError('context metadata length mismatch')
    valid = np.ones(original.shape, dtype=bool) if valid_vectors is None else np.asarray(valid_vectors, dtype=bool)
    if valid.shape != original.shape:
        raise ValueError('context vector mask length mismatch')
    valid = valid & np.isfinite(original)
    result = original.copy()
    if not weight:
        return result
    for index, score in enumerate(original):
        session = session_ids[index]
        if not valid[index] or not session or not str(session).strip():
            continue
        neighbor_score = score
        for neighbor in (index - 1, index + 1):
            if not 0 <= neighbor < len(original) or not valid[neighbor]:
                continue
            if (user_ids[neighbor] == user_ids[index] and session_ids[neighbor] == session
                    and abs(sequences[neighbor] - sequences[index]) == 1):
                neighbor_score = max(neighbor_score, original[neighbor])
        result[index] = (1 - weight) * score + weight * neighbor_score
    return result


def image_first_order(candidate_ids, records):
    """Records are in source order; sequence gaps prevent pairing.

    Caller supplies an explicit, unique candidate set. Only pairs with both
    members selected are reordered. Output preserves every selected ID.
    """
    ids=list(candidate_ids)
    if len(ids)!=len(set(ids)):
        raise ValueError('duplicate candidate IDs')
    lookup={r['id']:r for r in records}
    if len(lookup)!=len(records):
        raise ValueError('duplicate source IDs')
    if any(i not in lookup for i in ids):
        raise ValueError('unknown candidate ID')
    pairs={}
    for a,b in zip(records,records[1:]):
        if (a['role']=='user' and a['has_image'] and b['role']=='assistant'
                and a['session_id'] and str(a['session_id']).strip()
                and a['user_id']==b['user_id']
                and a['session_id']==b['session_id']
                and b['sequence']==a['sequence']+1):
            pairs[a['id']]=(a['id'],b['id'])
            pairs[b['id']]=(a['id'],b['id'])
    selected=set(ids);emitted=set();head=[];tail=[]
    for mid in ids:
        pair=pairs.get(mid)
        if pair and pair[0] in selected and pair[1] in selected:
            if pair[0] not in emitted:
                emitted.add(pair[0]);head.append(pair[0]);tail.append(pair[1])
        else:
            head.append(mid)
    result=head+tail
    assert len(result)==len(ids) and set(result)==selected
    return result
