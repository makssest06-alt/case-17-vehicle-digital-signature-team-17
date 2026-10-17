export HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
PY=C:/Users/user/PycharmProjects/AIOS/BM/case17_data/venv_gpu/Scripts/python.exe
for spec in "ctrl 0" "hm 0.5"; do set -- $spec
  s=$(date +%s)
  $PY ft_hardmine.py --init C:/Users/user/Downloads/clean_student_a10_ep28.pt --rows-csv clean_train_rows.csv --anchor a10_anchor.npy --images-dir C:/Users/user/PycharmProjects/AIOS/BM/case17_data/dataset/images --out-dir runs --tag $1 --mine $2 --workers 6 > runs/$1.log 2>&1
  echo "wall $(( $(date +%s) - s )) s" >> runs/$1.log
done
