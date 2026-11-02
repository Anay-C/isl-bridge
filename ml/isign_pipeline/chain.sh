#!/bin/bash
# chain.sh TAG "TRAIN FLAGS" seed...  Held-out recipe (CSLTR signers 0,3 never trained on), then eval_live on them.
SP="$(cd "$(dirname "$0")" && pwd)"
TAG=$1; FLAGS=$2; shift 2
cd "/c/Meta Hackathon/ml"
for s in "$@"; do
  out=../data/isl_exp/model_${TAG}_s$s.pt
  if [ ! -f "$out" ]; then
    ../.venv-ml/Scripts/python -u train.py --vocab --init ../data/model_all.pt --epochs 100 --mined-cap 0 \
      --seed $s --split-seed 0 $FLAGS --out $out > ../data/isl_exp/train_${TAG}_s$s.log 2>&1 \
      || { echo "TRAIN FAILED $TAG s$s"; tail -5 ../data/isl_exp/train_${TAG}_s$s.log; continue; }
  fi
  grep -E "iSign clips|classes \|" ../data/isl_exp/train_${TAG}_s$s.log | head -2
  tail -1 ../data/isl_exp/train_${TAG}_s$s.log
  ../.venv-ml/Scripts/python eval_live.py --model $out --signers 0,3 --thr 0.3 \
    --out ../data/isl_exp/eval_${TAG}_s$s.json > ../data/isl_exp/eval_${TAG}_s$s.txt 2>&1 &
done
wait
../.venv-ml/Scripts/python "$SP/summ.py" ../data/isl_exp/eval_v10pw_s*.json ../data/isl_exp/eval_${TAG}_s*.json
