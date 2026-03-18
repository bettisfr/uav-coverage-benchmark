from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class BinaryMetrics:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        den = self.tp + self.fp
        return self.tp / den if den else 0.0

    @property
    def recall(self) -> float:
        den = self.tp + self.fn
        return self.tp / den if den else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        den = p + r
        return (2 * p * r / den) if den else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total else 0.0

    def to_dict(self):
        out = asdict(self)
        out.update(
            {
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "accuracy": self.accuracy,
            }
        )
        return out
