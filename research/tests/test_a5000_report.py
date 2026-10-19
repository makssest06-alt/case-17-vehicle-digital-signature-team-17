"""research/a5000_report.py: таблица захода research/a5000_session.sh (a10 и L75 × EXTRACT_THREADS), порог и
формула отказа, остановка на шаге — по синтетическим файлам в OUT_DIR.

Запуск из корня репозитория:
    python -m unittest discover -s research/tests -v
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research import a5000_report as ar


def bench(lat: float, fps: float, vram: float) -> dict:
    thr = min(1.0, max(0.0, (fps - 50) / 50))
    lat_s = 1.0 if lat <= 40 else max(0.0, (80 - lat) / 40)
    return {"model": "m", "device": "cuda", "fp16": True, "latency_b1_ms": {"median": lat, "p95": lat + 2},
            "throughput_fps": {"32": fps}, "best_fps": fps, "best_batch": 32, "peak_vram_MB": vram,
            "weights_size_q37": {"MB": 362.1}, "latency_score": lat_s, "throughput_score": thr,
            "performance_pct_of_20": round(10 * lat_s + 10 * thr, 2)}


RECAL = {"presets": {"итог": {"balanced": 0.77, "strict": 0.93}}, "tau": 0.69,
         "rule": {"s_hi": 0.86, "t_iou": 0.6, "t_center": 0.3, "t_size": float("inf")},
         "metrics": {"итог": {"361": {"Q15": 0.9, "mAP@10": 0.91}, "1111": {"Q15": 0.8, "mAP@10": 0.8}}},
         "closed_gallery": {"большое": True}, "calibration": {"большое": True}}


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.out = Path(self.dir.name)
        (self.out / "session.cfg").write_text("models=a10 l75\nthreads=4 8\n", encoding="utf-8")
        (self.out / "env_host.txt").write_text("gpu=NVIDIA RTX A5000\ncpu_model=Intel Xeon\ncpu_threads=32\n",
                                               encoding="utf-8")

    def run_report(self) -> str:
        with patch("sys.argv", ["a5000_report.py", str(self.out)]), patch("builtins.print"):
            self.assertEqual(ar.main(), 0)
        return (self.out / "session.md").read_text(encoding="utf-8")

    def test_full_session_table(self):
        for m, t, b in (("a10", 4, bench(20.0, 95.0, 700)), ("a10", 8, bench(19.0, 120.0, 700)),
                        ("l75", 4, bench(45.0, 70.0, 2400))):
            (self.out / f"bench_{m}_t{t}.json").write_text(json.dumps(b), encoding="utf-8")
        (self.out / "bench_l75_t8.err").write_text("CUDA out of memory\n", encoding="utf-8")
        (self.out / "recalibrate_l75.json").write_text(json.dumps(RECAL), encoding="utf-8")
        (self.out / "bench_done").touch()
        (self.out / "status.txt").write_text("готово\n", encoding="utf-8")
        (self.out / "timings.txt").write_text("замеры=420\nпересчёт L75=200\n", encoding="utf-8")
        md = self.run_report()
        self.assertIn("| a10 | 8 | 19.00 (21.00) | 120.0 (32) | 700 | 362 | 20.00 | 0.7586 / 0.9302 | 0.881 / 0.756 |", md)
        self.assertIn("| l75 | 4 | 45.00 (47.00) | 70.0 (32) | 2400 | 362 | 12.75 | 0.77 / 0.93 | 0.900 / 0.800 |", md)
        self.assertIn("| l75 | 8 | ошибка: CUDA out of memory", md)
        self.assertIn("10 % × latency_score + 10 % × throughput_score", md)
        self.assertIn("2× Intel Xeon Gold 6338", md)
        self.assertIn("всего 10 мин", md)
        s = json.loads((self.out / "session.json").read_text(encoding="utf-8"))
        self.assertEqual(s["status"], "готово")
        self.assertEqual(len(s["rows"]), 4)
        self.assertNotIn("closed_gallery", s["recalibrate_l75"])            # тяжёлое — только в recalibrate_l75.json
        self.assertEqual(s["refusal"]["l75"]["threshold"], 0.77)

    def test_stop_before_bench_has_no_table(self):
        (self.out / "status.txt").write_text("остановлено на п. 3\nвидеокарта не видна из контейнера\n", encoding="utf-8")
        md = self.run_report()
        self.assertIn("Статус: **остановлено на п. 3**", md)
        self.assertIn("## Остановка\n\nвидеокарта не видна из контейнера", md)
        self.assertNotIn("| Модель |", md)


if __name__ == "__main__":
    unittest.main()
