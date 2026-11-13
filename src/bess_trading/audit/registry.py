"""SQLite-backed immutable experiment and decision registry."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DecisionAudit:
    experiment_id: str
    decision_time_utc: str
    delivery_start_utc: str
    market: str
    model_version: str
    config_hash: str
    input_hashes: dict[str, str]
    solver: str
    solver_status: str
    objective_eur: float
    decision: dict[str, float]


class ExperimentRegistry:
    """Append-only local registry sufficient to reconstruct every decision."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    git_commit TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    model_versions_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    audit_json TEXT NOT NULL,
                    audit_hash TEXT NOT NULL UNIQUE,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def create_experiment(
        self, config: dict[str, Any], git_commit: str, model_versions: dict[str, str]
    ) -> str:
        experiment_id = str(uuid.uuid4())
        config_json = json.dumps(config, sort_keys=True, default=str)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?)",
                (
                    experiment_id,
                    datetime.now(UTC).isoformat(),
                    git_commit,
                    config_json,
                    canonical_hash(config),
                    json.dumps(model_versions, sort_keys=True),
                ),
            )
        return experiment_id

    def record_decision(self, audit: DecisionAudit) -> str:
        payload = json.dumps(asdict(audit), sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO decisions (experiment_id, audit_json, audit_hash) VALUES (?, ?, ?)",
                (audit.experiment_id, payload, digest),
            )
        return digest

    def reconstruct(self, audit_hash: str) -> DecisionAudit:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT audit_json FROM decisions WHERE audit_hash = ?", (audit_hash,)
            ).fetchone()
        if row is None:
            raise KeyError(audit_hash)
        return DecisionAudit(**json.loads(row[0]))
