from spike4_embeddings.metrics import estimate_openai_cost, summarize_batch


def test_summarize_batch_ok():
    rows = [
        {"status": "ok", "dim": 3, "attempts": 1},
        {"status": "ok", "dim": 3, "attempts": 2},
    ]
    # add minimal required keys used in function
    for r in rows:
        r.setdefault("chunk_id", "c")
    summary = summarize_batch(rows)
    assert summary["rows"] == 2
    assert summary["rows_ok"] == 2
    assert summary["dim"] == 3
    assert 1.0 <= summary["avg_attempts"] <= 2.0


def test_summarize_batch_with_errors():
    rows = [
        {"status": "ok", "dim": 3, "attempts": 1, "chunk_id": "c1"},
        {"status": "error:RuntimeError", "dim": None, "attempts": 1, "chunk_id": "c2"},
    ]
    summary = summarize_batch(rows)
    assert summary["rows"] == 2
    assert summary["rows_ok"] == 1
    assert summary["dim"] == 3  # taken from ok row


def test_estimate_openai_cost():
    # 10k tokens at $0.02/1k → $0.20
    assert abs(estimate_openai_cost(10_000, 0.02) - 0.2) < 1e-6
