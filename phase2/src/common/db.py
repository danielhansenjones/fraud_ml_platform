from __future__ import annotations

from psycopg_pool import ConnectionPool


def make_pool(dsn: str, min_size: int = 1, max_size: int = 5) -> ConnectionPool:
    pool = ConnectionPool(dsn, min_size=min_size, max_size=max_size, open=False)
    pool.open()
    return pool
