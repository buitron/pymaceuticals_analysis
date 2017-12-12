import random
import matplotlib.pyplot as plt

markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'x', 'D', 'd']


def axis_gen(pivot_df):
    axis = [list(pivot_df[column]) for column in pivot_df.columns]
    return axis


def errorbar_gen(y_axis, yerr_axis, column, x_axis, sem_axis):
    plt_errorbars = [plt.errorbar(x_axis, y_axis[i], yerr=sem_axis[i], marker=random.choice(markers),
        mec='black', ms=8, ls='dashdot', capsize=6, label=column.columns[i]) for i in range(len(y_axis))]
    return plt_errorbars
