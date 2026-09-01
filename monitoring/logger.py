import pandas as pd
import csv
import os
from datetime import datetime, timezone

LOG_FILE_PATH = "./request_log.csv"

LOG_COLUMNS = [
    "timestamp", "question", "selected_model", "routing_reason",
    "input_tokens", "output_tokens", "estimated_cost", "latency_seconds",
    "success", "was_budget_downgraded",
]


def _ensure_log_file_exists():

    if not os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_COLUMNS)


def log_request(
    question: str,
    selected_model: str,
    routing_reason: str,
    input_tokens: int,
    output_tokens: int,
    estimated_cost: float,
    latency_seconds: float,
    success: bool,
    was_budget_downgraded: bool = False,
):
   
    _ensure_log_file_exists()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "selected_model": selected_model,
        "routing_reason": routing_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": estimated_cost,
        "latency_seconds": round(latency_seconds, 3),
        "success": success,
        "was_budget_downgraded": was_budget_downgraded,
    }

    with open(LOG_FILE_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerow(row)


def load_logs():

    if not os.path.exists(LOG_FILE_PATH):
        return pd.DataFrame(columns=LOG_COLUMNS)

    return pd.read_csv(LOG_FILE_PATH)
