import pandas as pd
from intersample import intersample
from makeplots import makeplots

filename="testdata.csv" #this will need to be replaced with a different read-in from a UI browser window or something similar
df = pd.read_csv(filename) #make dataframe
N = len(df.columns)//2 # number of samples 

xmin = 1 # define lower limit for probability density plots (PDPs) and kernel density estimates (KDEs) and all plots
xmax = 4000 #upper limit for PDPs and KDEs and all plots
xint = 1 # discretization interval for PDPs and KDEs only

#DZtools options
DZstats = 1
DZmds = 0
PlotDistributions = 1

if DZstats == 1:
    KSTestD,KuiperTestV,Similarity,Likeness,CrossCorrelation = intersample(df,xmin,xmax,xint)

if PlotDistributions == 1:
    fig,axs = makeplots(df,xmin,xmax,xint)