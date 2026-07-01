"""Set-trie helpers for subset queries over predicate bitmasks."""

from __future__ import annotations

from dataclasses import dataclass, field

from predicates import mask_to_ids


@dataclass
class _TrieNode:
    terminal: bool = False
    children: dict[int, "_TrieNode"] = field(default_factory=dict)


@dataclass
class SetTrie:
    """A prefix trie over sorted predicate ids.

    It supports exact-safe pruning queries of the form:
    `exists A in trie such that A subset C`.
    """

    root: _TrieNode = field(default_factory=_TrieNode)
    size: int = 0

    def insert(self, mask: int) -> None:
        node = self.root
        for predicate_id in mask_to_ids(mask):
            node = node.children.setdefault(predicate_id, _TrieNode())
        if not node.terminal:
            node.terminal = True
            self.size += 1

    def exists_subset(self, mask: int) -> bool:
        ids = mask_to_ids(mask)
        return _exists_subset_from(self.root, ids, 0)


def _exists_subset_from(node: _TrieNode, ids: tuple[int, ...], start: int) -> bool:
    if node.terminal:
        return True
    for child_id, child in node.children.items():
        next_pos = _find_at_or_after(ids, child_id, start)
        if next_pos is None:
            continue
        if _exists_subset_from(child, ids, next_pos + 1):
            return True
    return False


def _find_at_or_after(ids: tuple[int, ...], target: int, start: int) -> int | None:
    for pos in range(start, len(ids)):
        if ids[pos] == target:
            return pos
        if ids[pos] > target:
            return None
    return None
