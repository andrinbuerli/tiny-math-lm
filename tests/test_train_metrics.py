import csv

from tiny_math_lm.train.metrics import elapsed_seconds_through_step, truncate_csv_after_step


def test_elapsed_seconds_uses_checkpoint_aligned_metric_row(tmp_path):
    metrics = tmp_path / "metrics.csv"
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["optimizer_step", "wall_seconds"])
        writer.writeheader()
        writer.writerows(
            [
                {"optimizer_step": 1, "wall_seconds": 2.5},
                {"optimizer_step": 2, "wall_seconds": 5.0},
                {"optimizer_step": 3, "wall_seconds": 8.0},
            ]
        )

    assert elapsed_seconds_through_step(
        metrics, step_column="optimizer_step", step=2
    ) == 5.0
    assert elapsed_seconds_through_step(metrics, step_column="optimizer_step", step=0) == 0.0


def test_elapsed_seconds_survives_metric_truncation_for_resume(tmp_path):
    metrics = tmp_path / "metrics.csv"
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["update", "wall_seconds"])
        writer.writeheader()
        writer.writerows(
            [
                {"update": 500, "wall_seconds": 3376.0},
                {"update": 501, "wall_seconds": 8.0},
            ]
        )

    elapsed_before = elapsed_seconds_through_step(metrics, step_column="update", step=500)
    truncate_csv_after_step(metrics, step_column="update", step=500)

    assert elapsed_before == 3376.0
    assert elapsed_seconds_through_step(metrics, step_column="update", step=500) == 3376.0


def test_truncate_upgrades_mixed_rows_to_current_metric_schema(tmp_path):
    metrics = tmp_path / "metrics.csv"
    old_fields = ["update", "wall_seconds", "accuracy"]
    current_fields = ["update", "timing", "wall_seconds", "accuracy"]
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(old_fields)
        writer.writerow([1, 10.0, 0.5])
        writer.writerow([2, 20.0, 0.75])
        # These values were appended by a newer writer before the CSV header
        # was upgraded, so their positions follow current_fields.
        writer.writerow([3, 0.25, 30.0, 1.0])

    truncate_csv_after_step(
        metrics,
        step_column="update",
        step=3,
        fieldnames=current_fields,
    )

    with metrics.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == current_fields
    assert rows == [
        {"update": "1", "timing": "", "wall_seconds": "10.0", "accuracy": "0.5"},
        {"update": "2", "timing": "", "wall_seconds": "20.0", "accuracy": "0.75"},
        {"update": "3", "timing": "0.25", "wall_seconds": "30.0", "accuracy": "1.0"},
    ]
