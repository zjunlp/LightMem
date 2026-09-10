"""Regression coverage for offline update forwarding."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from lightmem.memory.lightmem import LightMemory


def test_offline_update_trigger_forwards_score_threshold():
    memory = LightMemory.__new__(LightMemory)
    memory.config = SimpleNamespace(index_strategy="embedding")
    memory.logger = Mock()

    with patch.object(
        LightMemory, "offline_update_all_entries", autospec=True
    ) as offline_update_all_entries:
        memory.offline_update([], offline_update_trigger=True)

    offline_update_all_entries.assert_called_once_with(
        memory,
        score_threshold=0.8,
    )
