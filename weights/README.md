Прод этой ветки — `clean_student_large_150_ep150.pt` (ViT-L/16@256, 1 273 134 679 байт): при сборке запекается в образ, `/opt/weights` (`api/Dockerfile`); в git не лежит — скачивается из GitHub Release `weights-l150` по `WEIGHTS_URL` по умолчанию.
sha256 `6937d1163e239bcfd8ee76fe00da12a9d60becc89563b8fc5a475dc72df13d52` — сверяется при сборке (`WEIGHTS_SHA256`) и при старте (`/health`).
`reid_current.pt` — прежняя head-only модель, только для истории замеров в `research/`, сервис его не берёт.
Базовый кандидат a10 (ViT-B@224, 362 МБ) — в ветке `glebmain` (Release `weights-a10`).
