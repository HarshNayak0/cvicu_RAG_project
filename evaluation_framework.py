# evaluation_logger.py
import csv
from datetime import datetime
from pathlib import Path

EVAL_LOG_PATH = Path("evaluation_logs.csv")

# Ensure CSV exists with headers
if not EVAL_LOG_PATH.exists():
    with EVAL_LOG_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "query",
                "answer",
                "relevance",
                "correctness",
                "completeness",
                "source_alignment",
                "notes",
            ]
        )


def log_evaluation(
    query, answer, relevance, correctness, completeness, source_alignment, notes=""
):
    timestamp = datetime.now().isoformat()
    with EVAL_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                timestamp,
                query,
                answer.replace("\n", " ").strip(),
                relevance,
                correctness,
                completeness,
                source_alignment,
                notes.strip(),
            ]
        )
    print("✅ Evaluation logged.")
