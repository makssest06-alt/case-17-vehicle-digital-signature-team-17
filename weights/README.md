Прод — `clean_student_a10_ep28.pt` (Git LFS, 379 734 230 байт): при сборке запекается в образ, `/opt/weights` (`api/Dockerfile`).
sha256 `93114c5923869d01e92cd70c2c421f783d05ebdac5652a1f2dc919aa040bf6ce` — сверяется при сборке (`WEIGHTS_SHA256`) и при старте (`/health`).
Остальное (`reid_current.pt` — прежняя head-only модель) — только для истории замеров в `research/`, сервис его не берёт.
