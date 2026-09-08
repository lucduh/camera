import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    name: str
    fields: tuple[str, ...]
    task_token: str = "<s_donut>"
    missing_token: str = "<missing>"
    max_target_length: int = 256
    max_new_tokens: int = 256
    duplicate_policy: str = "first"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A task name is required")
        if not self.fields:
            raise ValueError(f"Task {self.name!r} has no fields")
        if len(self.fields) != len(set(self.fields)):
            raise ValueError(f"Task {self.name!r} contains duplicate fields")
        if self.max_target_length < 1 or self.max_new_tokens < 1:
            raise ValueError("Target and generation lengths must be positive")
        if self.duplicate_policy not in {"first", "last"}:
            raise ValueError("duplicate_policy must be 'first' or 'last'")

    @property
    def structural_tokens(self) -> tuple[str, ...]:
        tokens = [self.task_token, self.missing_token]
        for field in self.fields:
            tokens.extend((self.open_token(field), self.close_token(field)))
        return tuple(tokens)

    @staticmethod
    def open_token(field: str) -> str:
        return f"<s_{field}>"

    @staticmethod
    def close_token(field: str) -> str:
        return f"</s_{field}>"


def load_task(path: str | Path) -> TaskSpec:
    path = Path(path)
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    return TaskSpec(
        name=raw["name"],
        fields=tuple(raw["fields"]),
        task_token=raw.get("task_token", "<s_donut>"),
        missing_token=raw.get("missing_token", "<missing>"),
        max_target_length=raw.get("max_target_length", 256),
        max_new_tokens=raw.get("max_new_tokens", 256),
        duplicate_policy=raw.get("duplicate_policy", "first"),
    )
