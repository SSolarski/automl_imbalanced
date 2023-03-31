from skopt import load
import matplotlib.pyplot as plt


def plot_best_brf_score(data_ids):
    """
    Plot the best score at each iteration for each dataset (from backup folder)
    Parameters
    ----------
    data_ids : list
        List of dataset ids to plot

    """

    for id in data_ids:
        # read the saved files

        # load the pipeline stats
        pipeline_stats = load(f'backup/one_hour/pipeline_stats_{id}.pkl')
        mean_scores = pipeline_stats["outer_fold_0"]["mean_scores"].tolist()

        # calculate best score at each iteration
        best_so_far = []
        for score in mean_scores:
            if len(best_so_far) == 0:
                best_so_far.append(score)
            else:
                best_so_far.append(max(score, best_so_far[-1]))

        # plot the best results at each iteration
        # plt.style.use('fivethirtyeight')
        plt.plot(best_so_far)
        plt.legend(data_ids, loc='lower right', fontsize='small')
        plt.ylabel('Best Balanced Accuracy')
        plt.xlabel('Iteration')
    plt.show()


def plot_best_score(data_ids):
    """
    Plot the best score at each iteration for each dataset (from backup folder)
    Parameters
    ----------
    data_ids : list
        List of dataset ids to plot

    """

    for id in data_ids:
        # read the saved files

        # load the pipeline stats
        pipeline_stats = load(f'backup/pipeline_stats_{id}.pkl')
        mean_scores = pipeline_stats["outer_fold_0"]["mean_scores"].tolist()

        # calculate best score at each iteration
        best_so_far = []
        for score in mean_scores:
            if len(best_so_far) == 0:
                best_so_far.append(score)
            else:
                best_so_far.append(max(score, best_so_far[-1]))

        # plot the best results at each iteration
        # plt.style.use('fivethirtyeight')
        plt.plot(best_so_far)
        plt.legend(data_ids, loc='lower right', fontsize='small')
        plt.ylabel('Best Balanced Accuracy')
        plt.xlabel('Iteration')
    plt.show()
