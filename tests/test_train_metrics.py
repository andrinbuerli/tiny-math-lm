import csv

from tinymath.train.metrics import elapsed_seconds_through_step, truncate_csv_after_step


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
