import matplotlib.pyplot as plt
import numpy as np

from .draw_utils import COLOR, LINE_STYLE

def draw_latency_plot(name, success_ret, precision_ret=None, bold_name=None, axis=[0, 1], save_path = 'figs/'):
    # success plot
    fig, ax = plt.subplots()
    ax.grid(b=True)
    ax.set_aspect(1)
    plt.xlabel('Permitted Latency')
    plt.ylabel('AUC')
    plt.title(r'\textbf{Success plots of LAE on %s}' % (name))
    plt.axis([0, 1]+axis)
    success = {}
    thresholds = np.arange(0.0, 1.01, 0.02)
    for tracker_name in success_ret.keys():
        value = success_ret[tracker_name]
        success[tracker_name] = np.mean(list(value.values()))
    for idx, (tracker_name, auc) in  \
            enumerate(sorted(success.items(), key=lambda x:x[1], reverse=True)):
        if tracker_name == bold_name:
            label = r"\textbf{[%.3f] %s}" % (auc, tracker_name)
        else:
            label = "[%.3f] " % (auc) + tracker_name
        value = [success_ret[tracker_name][str(round(eta,2))] for eta in thresholds]
        plt.plot(thresholds, value,
                color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
    ax.legend(loc='lower left', labelspacing=0.2)
    ax.autoscale(enable=True, axis='both', tight=True)
    xmin, xmax, ymin, ymax = plt.axis()
    ax.autoscale(enable=False)
    ymax += 0.03
    ymin = 0
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xticks(np.arange(xmin, xmax+0.01, 0.1))
    plt.yticks(np.arange(ymin, ymax, 0.1))
    ax.set_aspect((xmax - xmin)/(ymax-ymin))
    plt.show()
    plt.savefig('Success_plots_LAE_on_%s.svg' % (name),bbox_inches='tight')

    if precision_ret:
        # norm precision plot
        fig, ax = plt.subplots()
        ax.grid(b=True)
        ax.set_aspect(50)
        plt.xlabel('Permitted Latency')
        plt.ylabel('CLE')
        plt.title(r'\textbf{Precision plots of LAE on %s}' % (name))
        plt.axis([0, 1]+axis)
        precision = {}
        thresholds = np.arange(0.0, 1.01, 0.02)
        for tracker_name in precision_ret.keys():
            value = precision_ret[tracker_name]
            precision[tracker_name] = np.mean(list(value.values()))
        for idx, (tracker_name, pre) in \
                enumerate(sorted(precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [precision_ret[tracker_name][str(round(eta,2))] for eta in thresholds]
            plt.plot(thresholds, value,
                    color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
        ax.legend(loc='lower right', labelspacing=0.2)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        ymin = 0
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 5))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        plt.show()
        plt.savefig('Precision_plots_LAE_on_%s.svg' % (name),bbox_inches='tight')

    # # norm precision plot
    # if norm_precision_ret:
    #     fig, ax = plt.subplots()
    #     ax.grid(b=True)
    #     plt.xlabel('Location error threshold')
    #     plt.ylabel('Precision')
    #     if attr == 'ALL':
    #         plt.title(r'\textbf{Normalized Precision plots of OPE on %s}' % (name))
    #     else:
    #         plt.title(r'\textbf{Normalized Precision plots of OPE - %s}' % (attr))
    #     norm_precision = {}
    #     thresholds = np.arange(0, 51, 1) / 100
    #     for tracker_name in precision_ret.keys():
    #         value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
    #         norm_precision[tracker_name] = np.mean(value, axis=0)[20]
    #     for idx, (tracker_name, pre) in \
    #             enumerate(sorted(norm_precision.items(), key=lambda x:x[1], reverse=True)):
    #         if tracker_name == bold_name:
    #             label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
    #         else:
    #             label = "[%.3f] " % (pre) + tracker_name
    #         value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
    #         plt.plot(thresholds, np.mean(value, axis=0),
    #                 color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
    #     ax.legend(loc='lower right', labelspacing=0.2)
    #     ax.autoscale(enable=True, axis='both', tight=True)
    #     xmin, xmax, ymin, ymax = plt.axis()
    #     ax.autoscale(enable=False)
    #     ymax += 0.03
    #     ymin = 0
    #     plt.axis([xmin, xmax, ymin, ymax])
    #     plt.xticks(np.arange(xmin, xmax+0.01, 0.05))
    #     plt.yticks(np.arange(ymin, ymax, 0.1))
    #     ax.set_aspect((xmax - xmin)/(ymax-ymin))
    #     plt.show()
