import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..stats.distributions import ecdf, pdp

def makeplots(df,xmin,xmax,xint):
    
    x = np.arange(xmin,xmax+1,xint) # define discretized x axis
    
    N = len(df.columns)//2 # number of samples 
    
    fig, axs = plt.subplots(2,1,sharex=True) #set up figure, probably is a better/more efficient/prettier way to do this!
        
    # loop thru the individual samples to make all PDPs and plot all CDFs
    for j in range(N):
        data = df.iloc[:,j*2:j*2+2] #each sample is a set of 2 columns, so use this to loop thru and select pairs of columns
        data = data.apply(pd.to_numeric,errors='coerce')
        data = data.dropna() #remove N/A
        cdfx,cdfy = ecdf(data,xmin,xmax,xint) #make CDF
        PDP = pdp(data,xmin,xmax,xint) #make PDPs
        axs[0].plot(cdfx, cdfy) #plot CDFs on top of each other
        axs[1].plot(x, PDP)
    plt.xlabel('Age (Ma)')
    return(fig,axs)
