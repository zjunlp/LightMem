import pytest

from service.multimodal_events import event_neighbors, complete_event_ranking, merge_event_tail


def capture(stamp, lon=0, user='a'):
    return dict(user_id=user, capture_datetime_iso=stamp, gps_latitude=0, gps_longitude=lon)


def test_explicit_offsets_direct_edges_and_missing_metadata():
    records = [capture('2026-01-01T00:00:00+00:00'),
               capture('2026-01-01T01:00:00+01:00', .0007),
               capture('2026-01-01T00:00:00+00:00', .0014),
               capture('2026-01-01T00:00:00'),
               capture('2026-01-01T00:00:00+00:00', float('nan'))]
    # Adjacent distances are <100m; endpoints >100m must not become transitive neighbors.
    assert event_neighbors(records, user_id='a') == [[1], [0, 2], [1], [], []]


def test_user_boundary_rejected():
    with pytest.raises(ValueError, match='mixed user'):
        event_neighbors([capture('2026-01-01T00:00:00Z', user='b')], user_id='a')


def test_event_completion_is_bounded_and_deduplicated():
    order = list(range(8))
    scores = list(range(8, 0, -1))
    neighbors = [[7, 6], [6], [], [], [], [], [], []]
    completed = complete_event_ranking(order, scores, neighbors, head=2, limit=6)
    assert completed == [0, 1, 2, 6, 3, 7]
    assert merge_event_tail(order[:6], completed, head=2, additions=1) == [0, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match='outside scoped'):
        complete_event_ranking(order, scores, [[99]] * 8, head=2, limit=6)
