from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Deque, Dict, List

if TYPE_CHECKING:
    from unirl.types.sample import Sample


def root_of(sample: "Sample") -> str:
    if not sample.parts or not sample.parts[0].sample_ids:
        raise ValueError("trajectory has no root sample id")
    return sample.parts[0].sample_ids[0]


class PendingGroups:
    def __init__(self, group_size: int) -> None:
        self._group_size = int(group_size)
        if self._group_size <= 0:
            raise ValueError(f"group_size must be positive; got {group_size}")
        self._by_root: Dict[str, List["Sample"]] = {}

    def add(self, samples: List["Sample"]) -> List[List["Sample"]]:
        complete = []
        for sample in samples:
            root = root_of(sample)
            siblings = self._by_root.setdefault(root, [])
            siblings.append(sample)
            if len(siblings) > self._group_size:
                raise RuntimeError(f"root {root!r} has more than {self._group_size} terminal siblings")
            if len(siblings) == self._group_size:
                complete.append(self._by_root.pop(root))
        return complete

    def get(self, root: str) -> List["Sample"]:
        return list(self._by_root.get(root, []))

    def discard(self, root: str) -> int:
        return len(self._by_root.pop(root, []))

    def __len__(self) -> int:
        return len(self._by_root)


class CompletedGroups:
    def __init__(self) -> None:
        self._groups: Deque[List["Sample"]] = deque()

    def extend(self, groups: List[List["Sample"]]) -> None:
        self._groups.extend(groups)

    def prepend(self, groups: List[List["Sample"]]) -> None:
        self._groups.extendleft(reversed(groups))

    def popleft(self) -> List["Sample"]:
        return self._groups.popleft()

    def __len__(self) -> int:
        return len(self._groups)


__all__ = ["CompletedGroups", "PendingGroups", "root_of"]
