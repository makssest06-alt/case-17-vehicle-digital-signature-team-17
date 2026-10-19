Прод этой ветки — `soup150s2.pt` (ViT-L/16@256, model soup чекпоинтов l150 и s2_ep75, 1 273 096 203 байт): при сборке запекается в образ, `/opt/weights` (`api/Dockerfile`); в git не лежит — скачивается из GitHub Release `weights-soup150s2` по `WEIGHTS_URL` по умолчанию. Рецепт soup — `research/make_soup.py`.
sha256 `64e2f1897b593d9de7100fd6449058f0fad57f98de0bcbdfe4c546c5a6a21553` — сверяется при сборке (`WEIGHTS_SHA256`) и при старте (`/health`).
`reid_current.pt` — прежняя head-only модель, только для истории замеров в `research/`, сервис его не берёт.
Базовый кандидат a10 (ViT-B@224, 362 МБ) — в ветке `glebmain` (Release `weights-a10`).
