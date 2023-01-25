import os
import sys
import time
import argparse
import functools
import json
sys.path.append("./")

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from pysot.datasets import DatasetFactory
from pysot.evaluation import OPEBenchmark, LAEBenchmark
from pysot.visualization import draw_success_precision, draw_eao, draw_f1, draw_latency_plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Object Tracking Evaluation')
    parser.add_argument('--dataset_dir', type=str, help='dataset root directory')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--tracker_result_dir', type=str, default = '/path/to/results_rt', help='tracker result root')
    parser.add_argument('--trackers', nargs='+')
    parser.add_argument('--vis', dest='vis', action='store_true')
    parser.add_argument('--eta', type=float, default = 1.0, help='maximum permited latency for streaming evaluation, for eta=0.0, use strict policy')
    parser.add_argument('--show_video_level', dest='show_video_level', action='store_true')
    parser.add_argument('--num', type=int, help='number of processes to eval', default=1)
    args = parser.parse_args()

    tracker_dir = args.tracker_result_dir
    trackers = args.trackers
    root = args.dataset_dir

    assert len(trackers) > 0
    args.num = max(args.num, len(trackers))
    # DTB70
    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=args.dataset_dir)
    dataset.set_tracker(tracker_dir, trackers)
    if args.eta:
        benchmark = LAEBenchmark(dataset, args.eta)
    else:
        benchmark = OPEBenchmark(dataset)
    success_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
            trackers), desc='eval success', total=len(trackers), ncols=100):
            success_ret.update(ret)
    precision_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
            trackers), desc='eval precision', total=len(trackers), ncols=100):
            precision_ret.update(ret)
    if args.eta:
        tracker_auc, tracker_cle = benchmark.show_result(success_ret, precision_ret, show_video_level=args.show_video_level)
        auc_json_path = '{}_AUC.json'.format(args.dataset)
        cle_json_path = '{}_CLE.json'.format(args.dataset)
        with open(auc_json_path, 'w') as auc_f:
            json.dump(tracker_auc, auc_f)
        with open(cle_json_path, 'w') as cle_f:
            json.dump(tracker_cle, cle_f)
    else:
        benchmark.show_result(success_ret, precision_ret, show_video_level=args.show_video_level)

    if args.vis:
        if args.eta:
            draw_latency_plot(args.dataset, tracker_auc,
                            tracker_cle)
    
    
