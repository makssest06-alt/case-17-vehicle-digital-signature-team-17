"""REFUSAL_PRESETS в app/config.py совпадают с расчётом research/closed_gallery.py (протокол q17).

Нужен только app.config (без fastapi). Если research/ рядом нет (образ сервиса) — пропуск.
    python -m unittest tests.test_refusal_presets -v      # из каталога api/
"""
import json
import unittest
from pathlib import Path

RESULT = Path(__file__).resolve().parents[2] / "research" / "honest_eval_results" / "closed_gallery_soup150s2.json"


@unittest.skipUnless(RESULT.exists(), "нет research/honest_eval_results/closed_gallery_soup150s2.json")
class RefusalPresetsTest(unittest.TestCase):
    def test_presets_match_research(self):
        from app.config import REFUSAL_PRESETS

        res = json.loads(RESULT.read_text(encoding="utf-8"))
        computed = res["config_presets"]
        self.assertGreater(len(computed), 0)
        for key, got in REFUSAL_PRESETS.items():
            if key == (0, "off", "off", "raw"):
                continue                        # legacy-пресет до постобработки, калиброван под a10
            with self.subTest(key):
                if key[1:3] == ("off", "off"):  # «DBA»: прежний генератор не писал его в config_presets — по presets
                    p = res["presets" if key[3] == "raw" else "presets_alt"]["DBA"]
                    want = {"balanced": round(p["threshold"], 4), "strict": round(p["strict"], 4)}
                else:
                    want = computed[repr(key)]
                self.assertEqual((got["balanced"], got["strict"]), (want["balanced"], want["strict"]))

    def test_dedup_rule_matches_calibration(self):
        from app.config import Settings

        rule = json.loads(RESULT.read_text(encoding="utf-8"))["rules"]["ours"]
        s = Settings()
        self.assertEqual((s.pp_dedup_sim, s.pp_dedup_iou, s.pp_dedup_center, s.pp_dedup_size),
                         (rule["s_hi"], rule["t_iou"], rule["t_center"], rule["t_size"]))

    def test_current_service_point_untouched(self):
        from app.config import REFUSAL_PRESETS, Settings

        cur = REFUSAL_PRESETS[(1, "off", "off", "raw")]                   # production: «DBA»
        s = Settings()
        self.assertEqual((cur["balanced"], cur["strict"]), (s.search_threshold, s.search_threshold_strict))
        from app.pp import state
        self.assertEqual(state.preset_key(s), (1, "off", "off", "raw"))     # пресет по умолчанию — ключ «DBA»


if __name__ == "__main__":
    unittest.main()
