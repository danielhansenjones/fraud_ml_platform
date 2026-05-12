from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import psycopg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ARTIFACTS = Path(__file__).parent.parent.parent / "training" / "artifacts"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upsert_model(
    conn: psycopg.Connection,
    version: str,
    path: str,
    sha: str,
    role: str,
    notes: str,
) -> None:
    row = conn.execute(
        "SELECT role FROM models WHERE model_version = %s", (version,)
    ).fetchone()

    if row is None:
        conn.execute(
            "INSERT INTO models (model_version, model_path, onnx_sha256, role, notes) "
            "VALUES (%s, %s, %s, %s, %s)",
            (version, path, sha, role, notes),
        )
        log.info("inserted %s %s", role, version[:16])
        return

    current_role = row[0]
    if current_role == role:
        log.info("%s %s already current, skipping", role, version[:16])
        return

    if current_role == "retired":
        # Re-promoting a retired version masks an audit-trail incident: the
        # original retirement was a deliberate action; silently flipping it
        # back during a re-run of register_models is exactly the kind of
        # "blind idempotency" the senior review flagged. Force an explicit
        # path (manual UPDATE) instead.
        raise RuntimeError(
            f"refusing to promote retired model {version[:16]} to {role}; "
            "to re-promote, run an explicit UPDATE on the models table"
        )

    conn.execute(
        "UPDATE models SET role=%s, model_path=%s, onnx_sha256=%s, notes=%s "
        "WHERE model_version=%s",
        (role, path, sha, notes, version),
    )
    log.info("updated %s %s -> %s", current_role, version[:16], role)


def main() -> None:
    dsn = os.environ.get("POSTGRES_DSN", "postgresql://fraud:fraud@localhost:5432/fraud")

    champion_onnx = ARTIFACTS / "final_model.onnx"
    challenger_onnx = ARTIFACTS / "challenger" / "lgbm_final_model.onnx"

    if not champion_onnx.exists():
        raise FileNotFoundError(f"champion ONNX not found: {champion_onnx}")
    if not challenger_onnx.exists():
        raise FileNotFoundError(f"challenger ONNX not found: {challenger_onnx}")

    champ_sha = sha256(champion_onnx)
    chall_sha = sha256(challenger_onnx)

    champ_version = champ_sha
    chall_version = chall_sha

    try:
        conn = psycopg.connect(dsn)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to Postgres ({dsn}). "
            "Start it first: docker compose up -d postgres"
        ) from exc

    try:
        with conn.transaction():
            upsert_model(
                conn,
                champ_version,
                "/app/artifacts/final_model.onnx",
                champ_sha,
                "champion",
                "XGBoost v1, phase 1 baseline",
            )
            # Retire any older challenger row that is not the version we are
            # about to register. Skip if it is already retired.
            conn.execute(
                "UPDATE models SET role='retired', retired_at=NOW() "
                "WHERE role='challenger' AND model_version != %s",
                (chall_version,),
            )
            upsert_model(
                conn,
                chall_version,
                "/app/artifacts/challenger/lgbm_final_model.onnx",
                chall_sha,
                "challenger",
                "LightGBM v1, 200 Optuna trials",
            )
    finally:
        conn.close()

    versions = {"champion_version": champ_version, "challenger_version": chall_version}
    out = ARTIFACTS / "model_versions.json"
    out.write_text(json.dumps(versions, indent=2))

    log.info("champion registered: version=%s", champ_version)
    log.info("challenger registered: version=%s", chall_version)
    log.info("versions written to %s", out)


if __name__ == "__main__":
    main()
