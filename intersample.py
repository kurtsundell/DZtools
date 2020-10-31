import numpy as np
import pandas as pd
from measures import KSTest, KuiperTest, SimilarityTest, LikenessTest, CrossCorrelationTest
from distributions import pdp

def intersample(df,xmin,xmax,xint):
    x = np.arange(xmin,xmax+1,xint) # define discretized x axis
    N = len(df.columns)//2 # number of samples 
    PDPs = np.zeros((len(x),N)) # preallocate PDP variable

    # loop thru the individual samples to make all PDPs and plot all CDFs
    for j in range(N):
        data = df.iloc[:,j*2:j*2+2] #each sample is a set of 2 columns, so use this to loop thru and select pairs of columns
        data = data.apply(pd.to_numeric,errors='coerce')
        data = data.dropna() #remove N/A
        PDPs[:,j] = pdp(data,xmin,xmax,xint)
   
    KSTestD = np.zeros((N,N)) #Allocate pairwise comparison matrix for KS D statistic
    KuiperTestV = np.zeros((N,N)) #Allocate pairwise comparison matrix for Kuipet Test V statistic
    Similarity = np.zeros((N,N)) #Allocate pairwise comparison matrix for Similarity
    Likeness = np.zeros((N,N)) #Allocate pairwise comparison matrix for Likeness
    CrossCorrelation = np.zeros((N,N)) #Allocate pairwise comparison matrix for Cross-correlation
    
    #loop thru comparison of every sample vs every sample, including vs themselves
    for i in range(N): 
        for j in range(N):
            datai = df.iloc[:,i*2:i*2+2] #select sample 1
            datai = datai.apply(pd.to_numeric, errors='coerce')
            datai = datai.dropna()
            mi = datai.iloc[:,0]
            dataj = df.iloc[:,j*2:j*2+2] #select sample 2
            dataj = dataj.apply(pd.to_numeric, errors='coerce')
            dataj = dataj.dropna()    
            mj = dataj.iloc[:,0]
            KSTestD[i,j] = KSTest(mi,mj)
            if i == j:
                KSTestD[i,j] = 0
            KuiperTestV[i,j] = KuiperTest(mi,mj)
            if i == j:
                KuiperTestV[i,j] = 0
            Similarity[i,j] = SimilarityTest(PDPs[:,i],PDPs[:,j])
            Likeness[i,j] = LikenessTest(PDPs[:,i],PDPs[:,j])
            CrossCorrelation[i,j] = CrossCorrelationTest(PDPs[:,i],PDPs[:,j])
    return(KSTestD,KuiperTestV,Similarity,Likeness,CrossCorrelation)