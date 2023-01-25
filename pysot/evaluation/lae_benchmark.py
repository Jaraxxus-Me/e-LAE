import numpy as np

from colorama import Style, Fore

from ..utils import overlap_ratio, success_overlap, success_error

class LAEBenchmark:
    """
    Args:
        result_path: result path of your tracker
                should the same format like VOT
    """
    def __init__(self, dataset, eta):
        self.dataset = dataset
        self.eta = eta

    def convert_bb_to_center(self, bboxes):
        return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                         (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T

    def convert_bb_to_norm_center(self, bboxes, gt_wh):
        return self.convert_bb_to_center(bboxes) / (gt_wh+1e-16)

    def eval_success(self, eval_trackers=None):
        """
        Args: 
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        success_ret = {}
        for tracker_name in eval_trackers:
            success_ret_ = {}
            for eta in np.arange(0, self.eta+0.01, 0.02):
                eta = str(round(eta,2))
                success_ret__ = {}
                for video in self.dataset:
                    video.pred_trajs = {}
                    gt_traj = np.array(video.gt_traj)
                    if tracker_name not in video.pred_trajs:
                        tracker_traj = video.load_tracker(self.dataset.tracker_path,
                                tracker_name, eta, False)
                        tracker_traj = np.array(tracker_traj)
                    else:
                        tracker_traj = np.array(video.pred_trajs[tracker_name])
                    n_frame = len(gt_traj)
                    if hasattr(video, 'absent'):
                        gt_traj = gt_traj[video.absent == 1]
                        tracker_traj = tracker_traj[video.absent == 1]
                    success_ret__[video.name] = success_overlap(gt_traj, tracker_traj, n_frame)
                success_ret_[eta] = success_ret__
            success_ret[tracker_name] = success_ret_
        return success_ret

    def eval_precision(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        precision_ret = {}
        for tracker_name in eval_trackers:
            precision_ret_ = {}
            for eta in np.arange(0, self.eta+0.01, 0.02):
                eta = str(round(eta,2))
                precision_ret__ = {}
                for video in self.dataset:
                    video.pred_trajs = {}
                    gt_traj = np.array(video.gt_traj)
                    if tracker_name not in video.pred_trajs:
                        tracker_traj = video.load_tracker(self.dataset.tracker_path,
                                tracker_name, eta, False)
                        tracker_traj = np.array(tracker_traj)
                    else:
                        tracker_traj = np.array(video.pred_trajs[tracker_name])
                    n_frame = len(gt_traj)
                    if hasattr(video, 'absent'):
                        gt_traj = gt_traj[video.absent == 1]
                        tracker_traj = tracker_traj[video.absent == 1]
                    gt_center = self.convert_bb_to_center(gt_traj)
                    tracker_center = self.convert_bb_to_center(tracker_traj)
                    thresholds = np.arange(0, 51, 1)
                    precision_ret__[video.name] = success_error(gt_center, tracker_center,
                            thresholds, n_frame)
                precision_ret_[eta] = precision_ret__
            precision_ret[tracker_name] = precision_ret_
        return precision_ret

    def eval_norm_precision(self, eval_trackers=None):
        """
        Args:
            eval_trackers: list of tracker name or single tracker name
        Return:
            res: dict of results
        """
        if eval_trackers is None:
            eval_trackers = self.dataset.tracker_names
        if isinstance(eval_trackers, str):
            eval_trackers = [eval_trackers]

        norm_precision_ret = {}
        for tracker_name in eval_trackers:
            norm_precision_ret_ = {}
            for eta in np.arange(0, self.eta+0.01, 0.02):
                eta = str(round(eta,2))
                precision_ret__ = {}
                for video in self.dataset:
                    video.pred_trajs = {}
                    gt_traj = np.array(video.gt_traj)
                    if tracker_name not in video.pred_trajs:
                        tracker_traj = video.load_tracker(self.dataset.tracker_path, 
                                tracker_name, eta, False)
                        tracker_traj = np.array(tracker_traj)
                    else:
                        tracker_traj = np.array(video.pred_trajs[tracker_name])
                    n_frame = len(gt_traj)
                    if hasattr(video, 'absent'):
                        gt_traj = gt_traj[video.absent == 1]
                        tracker_traj = tracker_traj[video.absent == 1]
                    gt_center_norm = self.convert_bb_to_norm_center(gt_traj, gt_traj[:, 2:4])
                    tracker_center_norm = self.convert_bb_to_norm_center(tracker_traj, gt_traj[:, 2:4])
                    thresholds = np.arange(0, 51, 1) / 100
                    precision_ret__[video.name] = success_error(gt_center_norm,
                            tracker_center_norm, thresholds, n_frame)
                norm_precision_ret_[eta] = precision_ret__
            norm_precision_ret[tracker_name] = norm_precision_ret_
        return norm_precision_ret

    def show_result(self, success_ret, precision_ret=None,
            norm_precision_ret=None, show_video_level=False, helight_threshold=0.6):
        """pretty print result
        Args:
            result: returned dict from function eval
        """
        # sort tracker
        tracker_auc = {}
        for tracker_name in success_ret.keys():
            auc={}
            for eta in np.arange(0.0, self.eta+0.01, 0.02):
                eta = str(round(eta, 2))
                auc[eta] = np.mean(list(success_ret[tracker_name][eta].values()))
            tracker_auc[tracker_name] = auc
        tracker_auc_ = sorted(tracker_auc.items(),
                             key=lambda x:x[1]['0.0'],
                             reverse=True)[:20]
        tracker_names = [x[0] for x in tracker_auc_]


        tracker_name_len = max((max([len(x) for x in success_ret.keys()])+2), 12)
        header = ("|{:^"+str(tracker_name_len)+"}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|").format(
                "Tracker name", "Success_0", "Success_0.5", "Success_1.0", "Precision_0", "Precision_0.5", "Precision_1.0")
        formatter = "|{:^"+str(tracker_name_len)+"}|{:^12.3}|{:^12.3}|{:^12.3}|{:^12.3}|{:^12.3}|{:^12.3}|"
        print('-'*len(header))
        print(header)
        print('-'*len(header))
        tracker_cle = {}
        for tracker_name in tracker_names:
            # success = np.mean(list(success_ret[tracker_name].values()))
            success = tracker_auc[tracker_name]
            success_0 = success['0.0']
            success_5 = success['0.5']
            success_1 = success['1.0']
            if precision_ret is not None:
                precision = {}
                for eta in np.arange(0.0, self.eta+0.01, 0.02):
                    eta = str(round(eta, 2))
                    precision[eta] = np.mean(list(precision_ret[tracker_name][eta].values()), axis=0)[20]
                precision_0 = precision['0.0']
                precision_5 = precision['0.5']
                precision_1 = precision['1.0']
                tracker_cle[tracker_name] = precision
            else:
                precision = {}
                precision_0 = 0
                precision_5 = 0
                precision_1 = 0
            if norm_precision_ret is not None:
                norm_precision = {}
                for eta in np.arange(0.0, self.eta+0.01, 0.02):
                    eta = str(round(eta, 2))
                    norm_precision[eta] = np.mean(list(norm_precision_ret[tracker_name][eta].values()), axis=0)[20]
                norm_precision_0 = norm_precision['0.0']
                norm_precision_5 = norm_precision['0.5']
                norm_precision_1 = norm_precision['1.0']
            else:
                norm_precision = {}
                norm_precision_0 = 0
                norm_precision_5 = 0
                norm_precision_1 = 0
            print(formatter.format(tracker_name, success_0, success_5, success_1, precision_0, precision_5, precision_1))
        print('-'*len(header))
        return tracker_auc, tracker_cle

        
