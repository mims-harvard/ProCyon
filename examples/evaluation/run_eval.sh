#! /bin/bash
set -eux

source /path/to/ProCyon/.venv/bin/activate

python /path/to/ProCyon/scripts/run_eval_framework.py \
	--from_yaml eval_args.yml \
	2>&1 | tee log.txt
