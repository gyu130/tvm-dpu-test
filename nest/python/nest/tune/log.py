"""调优日志：JSONL，一行一个 trial，支持断点续跑与去重。"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrialRecord:
    entity: tuple[int, ...]
    config: dict
    cost: float | None
    error: str | None = None
    detail: dict = field(default_factory=dict)
    runner: str = ""
    timestamp: float = 0.0

    @property
    def ok(self) -> bool:
        return (
            self.error is None
            and self.cost is not None
            and math.isfinite(float(self.cost))
        )

    @property
    def sort_key(self) -> tuple:
        """可行解优先，其次按代价、entity 排序，保证确定性。"""
        if self.ok:
            return (0, float(self.cost), self.entity)
        return (1, math.inf, self.entity)

    def to_dict(self) -> dict:
        return {
            "entity": list(self.entity),
            "config": dict(self.config),
            "cost": None if self.cost is None else round(float(self.cost), 2),
            "error": self.error,
            "detail": dict(self.detail),
            "runner": self.runner,
            "timestamp": round(self.timestamp, 3),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrialRecord":
        return cls(
            entity=tuple(int(item) for item in data.get("entity", ())),
            config=dict(data.get("config", {})),
            cost=data.get("cost"),
            error=data.get("error"),
            detail=dict(data.get("detail", {})),
            runner=str(data.get("runner", "")),
            timestamp=float(data.get("timestamp", 0.0)),
        )


class TuneLog:
    """追加式调优日志；``path=None`` 时只在内存中保存。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._records: list[TrialRecord] = []
        if self.path is not None and self.path.exists():
            self._records = self._load_file(self.path)

    @staticmethod
    def _load_file(path: Path) -> list[TrialRecord]:
        records: list[TrialRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(TrialRecord.from_dict(json.loads(line)))
        return records

    @property
    def records(self) -> list[TrialRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def entities(self) -> set[tuple[int, ...]]:
        return {record.entity for record in self._records}

    def record_for(self, entity: tuple[int, ...]) -> TrialRecord | None:
        for record in reversed(self._records):
            if record.entity == entity:
                return record
        return None

    def append(self, record: TrialRecord) -> None:
        self._records.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )

    def best(self) -> TrialRecord | None:
        if not self._records:
            return None
        return min(self._records, key=lambda record: record.sort_key)

    def feasible(self) -> list[TrialRecord]:
        return [record for record in self._records if record.ok]

    def summary_text(self, limit: int = 10) -> str:
        feasible = sorted(self.feasible(), key=lambda record: record.sort_key)
        lines = [
            f"TuneLog: {len(self._records)} records, {len(feasible)} feasible"
            + (f", path={self.path}" if self.path else "")
        ]
        for record in feasible[:limit]:
            lines.append(
                f"  entity={record.entity} cost={record.cost:.0f} "
                f"config={record.config}"
            )
        return "\n".join(lines)


def make_record(
    entity: tuple[int, ...],
    config: dict,
    cost: float | None,
    *,
    error: str | None = None,
    detail: dict | None = None,
    runner: str = "",
) -> TrialRecord:
    return TrialRecord(
        entity=tuple(entity),
        config=dict(config),
        cost=cost,
        error=error,
        detail=dict(detail or {}),
        runner=runner,
        timestamp=time.time(),
    )


__all__ = ["TrialRecord", "TuneLog", "make_record"]
