"""Thread-safe in-memory LRU cache with pinning support."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterator
from typing import Generic, TypeVar

K = TypeVar('K')
V = TypeVar('V')


class LRUCache(Generic[K, V]):
    """A bounded LRU cache.

    ``put`` returns the keys that were evicted to make room. Pinned entries are
    never selected for eviction; if every entry is pinned the cache is allowed
    to temporarily exceed its capacity rather than discard live state.

    Reads always update recency, including reads of pinned entries. Pinning only
    protects an entry from eviction while an operation is in flight; it does not
    freeze its position in the LRU order. A pinned entry's read may promote it
    past everything that existed when the pin started, but not past entries that
    were admitted afterwards: admits are later writes and therefore strictly
    more recent than work performed inside the already-running pinned turn.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError('cache capacity must be at least 1')
        self._capacity = capacity
        self._data: OrderedDict[K, V] = OrderedDict()
        self._pinned: set[K] = set()
        self._pin_seq: dict[K, int] = {}
        self._insert_seq: dict[K, int] = {}
        self._seq = 0
        self._lock = threading.RLock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._data

    def __iter__(self) -> Iterator[K]:
        with self._lock:
            return iter(list(self._data.keys()))

    def keys(self) -> list[K]:
        with self._lock:
            return list(self._data.keys())

    def get(self, key: K, default: V | None = None) -> V | None:
        with self._lock:
            if key not in self._data:
                return default
            self._promote(key)
            return self._data[key]

    def peek(self, key: K, default: V | None = None) -> V | None:
        with self._lock:
            return self._data.get(key, default)

    def insert(self, key: K, value: V) -> None:
        """Insert or overwrite without evicting anything.

        Owners that must control eviction themselves (for example to persist a
        victim first) use this instead of :meth:`put`; the cache may exceed its
        capacity temporarily rather than silently discard state.
        """
        with self._lock:
            self._store(key, value)

    def put(self, key: K, value: V) -> list[K]:
        with self._lock:
            if key in self._data:
                self._data[key] = value
                self._data.move_to_end(key)
                return []
            self._store(key, value)
            evicted: list[K] = []
            while len(self._data) > self._capacity:
                victim = self._oldest_unpinned()
                if victim is None:
                    break
                self._delete(victim)
                evicted.append(victim)
            return evicted

    def _store(self, key: K, value: V) -> None:
        if key not in self._insert_seq:
            self._seq += 1
            self._insert_seq[key] = self._seq
        self._data[key] = value
        self._data.move_to_end(key)

    def _promote(self, key: K) -> None:
        cutoff = self._pin_seq.get(key)
        if cutoff is None:
            self._data.move_to_end(key)
            return
        # A pinned entry cannot jump over entries admitted after the pin.
        items = [(item_key, item) for item_key, item in self._data.items() if item_key != key]
        value = self._data[key]
        position = len(items)
        for index, (item_key, _) in enumerate(items):
            if self._insert_seq.get(item_key, 0) > cutoff:
                position = index
                break
        items.insert(position, (key, value))
        self._data = OrderedDict(items)

    def _oldest_unpinned(self) -> K | None:
        for key in self._data:
            if key not in self._pinned:
                return key
        return None

    def lru_key(self) -> K | None:
        """Return the oldest non-pinned key without changing recency."""
        with self._lock:
            return self._oldest_unpinned()

    def _delete(self, key: K) -> None:
        del self._data[key]
        self._pinned.discard(key)
        self._pin_seq.pop(key, None)
        self._insert_seq.pop(key, None)

    def delete(self, key: K) -> bool:
        with self._lock:
            if key not in self._data:
                return False
            self._delete(key)
            return True

    def pin(self, key: K) -> None:
        with self._lock:
            if key not in self._data:
                raise KeyError(key)
            self._pinned.add(key)
            self._pin_seq[key] = self._seq

    def unpin(self, key: K) -> None:
        with self._lock:
            if key not in self._data:
                raise KeyError(key)
            self._pinned.discard(key)
            self._pin_seq.pop(key, None)

    def is_pinned(self, key: K) -> bool:
        with self._lock:
            return key in self._pinned
