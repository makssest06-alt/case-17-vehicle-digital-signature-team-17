"""REFUSAL_PRESETS в app/config.py совпадают с расчётом research/closed_gallery.py (протокол q17).

Нужен только app.config (без fastapi). Если research/ рядом нет (образ сервиса) — пропуск.
    python -m unittest tests.test_refusal_presets -v      # из каталога api/
"""
import ast
import json
import unittest
from pathlib import Path

RESULT = Path(__file__).resolve().parents[2] / "research" / "honest_eval_results" / "closed_gallery_a10_q17.json"


@unittest.skipUnless(RESULT.exists(), "нет research/honest_eval_results/closed_gallery_a10_q17.json")
class RefusalPresetsTest(unittest.TestCase):
    def test_presets_match_research(self):
        from app.config import REFUSAL_PRESETS

        computed = json.loads(RESULT.read_text(encoding="utf-8"))["config_presets"]
        self.assertGreater(len(computed), 0)
        for key, want in computed.items():
            with self.subTest(key):
                got = REFUSAL_PRESETS[ast.literal_eval(key)]
                self.assertEqual((got["balanced"], got["strict"]), (want["balanced"], want["strict"]))

    def test_dedup_rule_matches_calibration(self):
        from app.config import Settings

        path = RESULT.parent / "dedup_compare_a10.json"
        if not path.exists():
            self.skipTest("нет dedup_compare_a10.json")
        rule = json.loads(path.read_text(encoding="utf-8"))["rule"]
        s = Settings()
        self.assertEqual((s.pp_dedup_sim, s.pp_dedup_iou, s.pp_dedup_center, s.pp_dedup_size),
                         (rule["s_hi"], rule["t_iou"], rule["t_center"], rule["t_size"]))

    def test_current_service_point_untouched(self):
        from app.config import REFUSAL_PRESETS, Settings

        cur = REFUSAL_PRESETS[(1, "off", "both", "raw")]                  # production: итог
        s = Settings()
        self.assertEqual((cur["balanced"], cur["strict"]), (s.search_threshold, s.search_threshold_strict))
        from app.pp import state
        self.assertEqual(state.preset_key(s), (1, "off", "both", "raw"))    # пресет по умолчанию — ключ итога


if __name__ == "__main__":
    unittest.main()
