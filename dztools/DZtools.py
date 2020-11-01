import sys
#import argparse

import pandas as pd
import matplotlib.pyplot as plt

from dztools.stats.intersample import intersample
from dztools.utils.makeplots import makeplots

xmin = 1 # define lower limit for probability density plots (PDPs) and kernel density estimates (KDEs) and all plots
xmax = 4000 #upper limit for PDPs and KDEs and all plots
xint = 1 # discretization interval for PDPs and KDEs only

#DZtools options
DZstats = 1
DZmds = 0
PlotDistributions = 1

def DZ_main():
    filename = sys.argv[1]

    df = pd.read_csv(filename)

    if DZstats == 1:
        intersample_results = intersample(df, xmin, xmax, xint)
        print(intersample_results)

    if PlotDistributions == 1:
        fig, axs = makeplots(df, xmin, xmax, xint)
        plt.show()


if __name__ == '__main__':
    DZ_main()

