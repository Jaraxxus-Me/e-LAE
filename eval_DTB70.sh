#!/usr/bin/env bash
python bin/eval.py --dataset_dir /path/to/DTB70  \ # dataset path
--dataset DTB70 \ # dataset
--tracker_result_dir /path/to/results_rt/DTB70 \ # result path
--eta 0.0 \
--trackers RPN_Mob_M RPN_Mob_V RPN_Mob_MV