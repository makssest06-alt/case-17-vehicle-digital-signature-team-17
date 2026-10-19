"""Флаги постобработки выключены — сервис отвечает байт в байт как до неё.

Эталон api/tests/data/golden_pp_off.json снят на коде origin/glebmain 598c311
(api/tests/make_golden_pp_off.py). Сравниваются статус, заголовки и тело каждого
ответа и все вызовы базы: /v1/search (принятие, отказ, пустая галерея, плохой
bbox), /v1/gallery, bulk-import, DELETE, /v1/config.

Запуск из каталога api/:
    python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import service_fakes as sf  # noqa: E402

OFF = {"pp_dba_k": 0, "pp_camera_penalty": "off", "pp_dedup": "off", "pp_dedup_rank": "off",
       "pp_dedup_refusal": "off", "refusal_score": "raw", "search_threshold": 0.8025, "search_threshold_strict": 0.8465}


class PostprocessOffTest(unittest.TestCase):
    def test_defaults_match_production(self):
        from app.config import Settings
        from app.pp import state

        s = Settings()
        self.assertTrue(state.enabled(s))
        self.assertEqual((s.pp_dba_k, state.dedup_rank(s), state.dedup_refusal(s), s.refusal_score,
                          s.search_threshold, s.search_threshold_strict),
                         (1, "off", "both", "raw", 0.8232, 0.943))

    def test_empty_compose_variables_mean_config_defaults(self):
        """docker-compose.yml пробрасывает ${X:-}: пустая строка — значение из config.py."""
        import importlib
        import os
        from unittest.mock import patch

        from app import config

        empty = {k: "" for k in ("SEARCH_THRESHOLD", "SEARCH_THRESHOLD_STRICT", "REFUSAL_SCORE", "PP_DBA_K",
                                 "PP_CAMERA_PENALTY", "PP_CAMERA_TAU", "PP_CAMERA_TOPK", "PP_DEDUP",
                                 "PP_DEDUP_RANK", "PP_DEDUP_REFUSAL", "PP_DEDUP_SIM_ANY")}
        try:
            with patch.dict(os.environ, empty):
                s = importlib.reload(config).Settings()
        finally:
            importlib.reload(config)
        self.assertEqual((s.search_threshold, s.search_threshold_strict, s.refusal_score, s.pp_dba_k,
                          s.pp_camera_penalty, s.pp_dedup, s.pp_dedup_rank, s.pp_dedup_refusal),
                         (0.8232, 0.943, "raw", 1, "on", "off", "off", "both"))

    def test_responses_and_db_calls_match_golden(self):
        golden = json.loads((HERE / "data" / "golden_pp_off.json").read_text(encoding="utf-8"))
        main_module = sf.load_app()
        main_module.app.state.pp = None
        with sf.data_dir() as root:
            got = sf.run_requests(main_module.app, main_module, sf.FakeDB(), root, OFF)
        self.assertEqual([g["name"] for g in got], [g["name"] for g in golden])
        for g, e in zip(got, golden):
            with self.subTest(e["name"]):
                self.assertEqual(g["status"], e["status"])
                self.assertEqual(g["headers"], e["headers"])
                self.assertEqual(g["body"], e["body"])
                self.assertEqual(g["db_calls"], e["db_calls"])


if __name__ == "__main__":
    unittest.main()
