"""DBA галереи — реализация живёт в сервисе (api/app/pp/dba.py), здесь только импорт.

Так research/bigallery_pp.py, research/cheap_bg.py и тесты мерят ровно тот код,
который работает в сервисе.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
from app.pp.dba import WEIGHTS, DBAGallery, _weights, dba, dba_vectors, l2, neighbours  # noqa: E402,F401
