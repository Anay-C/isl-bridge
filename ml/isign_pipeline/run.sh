#!/bin/bash
# iSign end to end. First accept the conditions on https://huggingface.co/datasets/Exploration-Lab/iSign (HF_TOKEN in .env).
#   bash ml/isign_pipeline/run.sh > data/isign_pipeline.log 2>&1
# plan -> focused plan (hospital / low-data signs first) -> download + concurrent keypoint extraction -> mine with the
# held-out model -> train 3 seeds (--no-islvt) -> eval vs eval_v10pw_s*.json. Never touches data/model.pt.
SP="$(cd "$(dirname "$0")" && pwd)"
cd "$SP/../.."
PY=.venv-ml/Scripts/python
$PY -u "$SP/poll_access.py" || exit 1
[ -f data/isign/plan.json ] || $PY -u ml/fetch_isign.py --plan --per-sign 40 || exit 1
$PY -u "$SP/focus_plan.py" || exit 1
mkdir -p data/isign/videos data/isign/keypoints
$PY -u ml/extract.py --videos data/isign/videos --keypoints data/isign/keypoints --workers 14 --watch > data/isign_extract.log 2>&1 &
EX=$!
$PY -u "$SP/download_focus.py" > data/isign_download.log 2>&1
kill $EX 2>/dev/null; sleep 5; taskkill //F //T //PID $EX >/dev/null 2>&1
$PY -u ml/extract.py --videos data/isign/videos --keypoints data/isign/keypoints --workers 14 >> data/isign_extract.log 2>&1
$PY -u ml/mine_isign.py --model data/model_v10.pt --cmin 0.6 > data/isign_mine.log 2>&1
tail -6 data/isign_mine.log
bash "$SP/chain.sh" isign "--no-islvt" 0 1 2
