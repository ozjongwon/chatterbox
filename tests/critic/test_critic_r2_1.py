# REQ-4 regression: reads SHALL update recency, including reads of pinned
# conversations.  The cache deliberately skips move_to_end for pinned keys, so
# once the pin is released the pinned key is evicted before more-recently-used
# unpinned entries, violating normal LRU order.

from __future__ import annotations

from conversational_agent.cache import LRUCache


def test_pinned_read_still_updates_recency() -> None:
    cache = LRUCache(capacity=3)
    cache.put('A', 'a')
    cache.put('B', 'b')
    cache.put('C', 'c')

    # A is in flight, so it is pinned; a read of A should still make A the
    # most recently used conversation.
    cache.pin('A')
    cache.get('A')

    assert cache.put('D', 'd') == ['B']

    # Now that A is unpinned, C is the least recently used entry if A's read
    # updated recency.  The current code does not promote A while pinned, so
    # it wrongly evicts A here.
    cache.unpin('A')
    assert cache.put('E', 'e') == ['C']
