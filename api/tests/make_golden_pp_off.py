"""Эталон ответов сервиса до постобработки: api/tests/data/golden_pp_off.json.

Снят на коде origin/glebmain 598c311 (до флагов PP_*), запуск из каталога api/
этого коммита:
    DATA_DIR=<пусто> python tests/make_golden_pp_off.py
Тест test_pp_off.py сравнивает с ним ответы нового кода при выключенных флагах.
Перегенерировать только на коде без постобработки — иначе тест теряет смысл.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import service_fakes as sf  # noqa: E402


def main() -> int:
    with sf.data_dir() as root:
        os.environ["DATA_DIR"] = str(root)
        main_module = sf.load_app()
        out = sf.run_requests(main_module.app, main_module, sf.FakeDB(), root)
    path = HERE / "data" / "golden_pp_off.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"-> {path}: {len(out)} ответов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
