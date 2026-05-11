from __future__ import annotations

# Columns excluded from the model feature set. `TransactionID` is the row key,
# `TransactionDT` and `day` are temporal markers used for splitting (and would
# leak time directly into the model), `isFraud` is the target.
EXCLUDE_FROM_FEATURES: frozenset[str] = frozenset(
    {"TransactionID", "TransactionDT", "isFraud", "day"}
)
