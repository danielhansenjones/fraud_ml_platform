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

    champ_version = champ_sha[:16]
    chall_version = chall_sha[:16]

    try:
        conn = psycopg.connect(dsn)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to Postgres ({dsn}). "
            "Start it first: docker compose up -d postgres"
        ) from exc
    try:
        with conn.transaction():
            conn.execute(
                """
                INSERT INTO models (model_version, model_path, onnx_sha256, role, notes)
                VALUES (%s, %s, %s, 'champion', 'XGBoost v1, phase 1 baseline')
                ON CONFLICT (model_version) DO UPDATE SET
                    role = EXCLUDED.role,
                    onnx_sha256 = EXCLUDED.onnx_sha256,
                    model_path = EXCLUDED.model_path,
                    notes = EXCLUDED.notes
                """,
                (champ_version, "/app/artifacts/final_model.onnx", champ_sha),
            )
            # Only one row may carry role='challenger' at a time.
            conn.execute(
                "UPDATE models SET role='retired', retired_at=NOW() WHERE role='challenger' AND model_version != %s",
                (chall_version,),
            )
            conn.execute(
                """
                INSERT INTO models (model_version, model_path, onnx_sha256, role, notes)
                VALUES (%s, %s, %s, 'challenger', 'LightGBM v1, 200 Optuna trials')
                ON CONFLICT (model_version) DO UPDATE SET
                    role = EXCLUDED.role,
                    onnx_sha256 = EXCLUDED.onnx_sha256,
                    model_path = EXCLUDED.model_path,
                    notes = EXCLUDED.notes
                """,
                (chall_version, "/app/artifacts/challenger/lgbm_final_model.onnx", chall_sha),
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
