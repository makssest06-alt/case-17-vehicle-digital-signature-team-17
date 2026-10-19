"""research/a5000_session.sh, ensure_a10_weights: откуда сборка возьмёт веса a10, если в weights/ указатель LFS.

Функция вырезается из скрипта и запускается в bash во временной папке с заглушками say/stop.

Запуск из корня репозитория:
    python -m unittest discover -s research/tests -v
"""
import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "a5000_session.sh"
POINTER = "version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 379734230\n"
URL = "https://example.org/releases/download/weights-a10/clean_student_a10_ep28.pt"


def function_source() -> str:
    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    start = next(i for i, s in enumerate(lines) if s.startswith("ensure_a10_weights() {"))
    end = next(i for i in range(start, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start:end + 1])


class EnsureA10WeightsTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        (self.dir / "weights").mkdir()
        (self.dir / "api").mkdir()
        self.real = b"\x80" * (1 << 20) + b"a10"
        (self.dir / "api" / "Dockerfile").write_text(
            f"ARG WEIGHTS_SHA256={hashlib.sha256(self.real).hexdigest()}\n", encoding="utf-8")
        self.compose(URL)
        (self.dir / "weights" / "clean_student_a10_ep28.pt").write_text(POINTER, encoding="utf-8")

    def compose(self, default: str) -> None:
        (self.dir / "docker-compose.yml").write_text(
            "    build:\n      args:\n"
            f"        WEIGHTS_URL: ${{WEIGHTS_URL:-{default}}}   # по умолчанию — ассет GitHub Release\n",
            encoding="utf-8")

    def run_fn(self, **env):
        script = "\n".join([
            "set -euo pipefail",
            'say() { echo "SAY $*"; }',
            'stop() { echo "STOP $*"; exit 3; }',
            "sha256() { shasum -a 256 \"$1\" | cut -d' ' -f1; }",
            'A10_FILE="weights/clean_student_a10_ep28.pt"',
            function_source(),
            "ensure_a10_weights",
            'echo "RC0"',
        ])
        clean = {k: v for k, v in os.environ.items() if k not in ("WEIGHTS_URL", "A10_WEIGHTS")}
        return subprocess.run(["bash", "-c", script], cwd=self.dir, env={**clean, **env},
                              capture_output=True, text=True, timeout=60)

    def weights(self) -> bytes:
        return (self.dir / "weights" / "clean_student_a10_ep28.pt").read_bytes()

    def test_pointer_uses_compose_default(self):
        r = self.run_fn()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("RC0", r.stdout)
        self.assertIn(URL, r.stdout)
        self.assertEqual(self.weights(), POINTER.encode())

    def test_pointer_without_any_source_stops(self):
        self.compose("")
        r = self.run_fn()
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn("указатель LFS", r.stdout)

    def test_explicit_url_returns_quietly(self):
        r = self.run_fn(WEIGHTS_URL="https://example.org/other.pt")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("SAY", r.stdout)

    def test_a10_weights_wins_over_default_url(self):
        f = self.dir / "a10.pt"
        f.write_bytes(self.real)
        r = self.run_fn(A10_WEIGHTS=str(f))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.weights(), self.real)

    def test_a10_weights_with_wrong_sha_stops(self):
        f = self.dir / "a10.pt"
        f.write_bytes(self.real + b"x")
        r = self.run_fn(A10_WEIGHTS=str(f))
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertIn("sha256", r.stdout)
        self.assertEqual(self.weights(), POINTER.encode())

    def test_real_file_is_kept(self):
        (self.dir / "weights" / "clean_student_a10_ep28.pt").write_bytes(self.real)
        r = self.run_fn()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("SAY", r.stdout)
        self.assertEqual(self.weights(), self.real)


class DefaultUrlTests(unittest.TestCase):
    """Ссылка по умолчанию записана дважды — в docker-compose.yml и в api/Dockerfile; копии не должны разойтись."""

    def test_compose_and_dockerfile_agree(self):
        repo = SCRIPT.parents[1]
        r = subprocess.run(["bash", "-c", "sed -n 's/.*WEIGHTS_URL: \\${WEIGHTS_URL:-\\([^}]*\\)}.*/\\1/p' "
                            "docker-compose.yml | head -1"], cwd=repo, capture_output=True, text=True, timeout=30)
        compose_url = r.stdout.strip()
        docker_url = next(s.split("=", 1)[1].strip().strip('"')
                          for s in (repo / "api" / "Dockerfile").read_text(encoding="utf-8").splitlines()
                          if s.startswith("ARG WEIGHTS_URL="))
        self.assertTrue(compose_url, "в docker-compose.yml нет ссылки по умолчанию")
        self.assertEqual(compose_url, docker_url)
        self.assertIn("/releases/download/weights-a10/clean_student_a10_ep28.pt", compose_url)


if __name__ == "__main__":
    unittest.main()
