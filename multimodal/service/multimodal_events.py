"""Bounded event-neighbor completion over caller-scoped retrieval records."""
from __future__ import annotations

import math
import datetime


def event_neighbors(records, *, user_id, seconds=3600, metres=100):
    """Build direct capture neighbors, requiring explicit timezone and valid GPS.

    Missing or malformed capture fields simply disable that record's edges.
    Reject mixed-user galleries before any event expansion.
    """
    if not all(r.get('user_id') == user_id for r in records):
        raise ValueError('mixed user gallery')
    if not all(math.isfinite(x) and x >= 0 for x in (seconds, metres)):
        raise ValueError('invalid event bounds')
    captures = []
    for record in records:
        try:
            stamp = datetime.datetime.fromisoformat(record['capture_datetime_iso'])
            lat, lon = float(record['gps_latitude']), float(record['gps_longitude'])
            if (stamp.utcoffset() is None or not all(map(math.isfinite, (lat, lon)))
                    or not -90 <= lat <= 90 or not -180 <= lon <= 180):
                raise ValueError('invalid capture')
            captures.append((stamp.timestamp(), math.radians(lat), math.radians(lon)))
        except (KeyError, ValueError, TypeError, OverflowError):
            captures.append(None)
    result = [[] for _ in records]
    valid = sorted((capture[0], i) for i, capture in enumerate(captures) if capture is not None)
    for pos, (stamp, i) in enumerate(valid):
        _, lat, lon = captures[i]
        for other, j in valid[pos + 1:]:
            if other - stamp > seconds:
                break
            _, lat2, lon2 = captures[j]
            hav = math.sin((lat2-lat)/2)**2 + math.cos(lat)*math.cos(lat2)*math.sin((lon2-lon)/2)**2
            distance = 6371000 * 2 * math.asin(math.sqrt(min(1, max(0, hav))))
            if distance <= metres:
                result[i].append(j)
                result[j].append(i)
    return [sorted(items) for items in result]


def complete_event_ranking(order, scores, neighbors, *, limit=100, head=20):
    """Alternate original relevance order and direct neighbors of the head.

    The caller supplies neighbors from one user's gallery. No transitive event
    expansion is performed. Stable gallery indices break relevance ties.
    """
    if not 0 <= head <= limit <= len(order):
        raise ValueError('invalid ranking bounds')
    if len(set(order)) != len(order):
        raise ValueError('duplicate ranking IDs')
    universe = set(order)
    if any(not math.isfinite(float(scores[i])) for i in order):
        raise ValueError('non-finite relevance')
    lists = []
    for seed in order[:head]:
        adjacent = list(neighbors[seed])
        if any(i not in universe for i in adjacent):
            raise ValueError('neighbor outside scoped gallery')
        lists.append(sorted(adjacent, key=lambda i: (-float(scores[i]), i)))
    queue = [items[depth]
             for depth in range(max(map(len, lists), default=0))
             for items in lists if depth < len(items)]
    selected = list(order[:head])
    seen = set(selected)
    streams = (iter(order[head:]), iter(queue))
    while len(selected) < limit:
        for stream in streams:
            for item in stream:
                if item not in seen:
                    selected.append(item)
                    seen.add(item)
                    break
            if len(selected) == limit:
                break
    return selected


def merge_event_tail(baseline, completed, *, head=20, additions=5):
    """Keep the baseline head; insert a bounded event tail before its remainder."""
    if not 0 <= head <= len(baseline) or additions < 0:
        raise ValueError('invalid merge bounds')
    if len(set(baseline)) != len(baseline):
        raise ValueError('duplicate baseline IDs')
    prefix = list(baseline[:head])
    excluded = set(prefix)
    extras = []
    for item in completed:
        if item not in excluded and len(extras) < additions:
            extras.append(item)
            excluded.add(item)
    return (prefix + extras + [i for i in baseline[head:] if i not in excluded])[:len(baseline)]
