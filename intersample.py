import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename="testdata.csv" #this will need to be replaced with a different read-in from a UI browser window or something similar
df = pd.read_csv(filename) #make dataframe

N = len(df.columns)//2 # number of samples 

xmin = 1 # define lower limit for probability density plots (PDPs) and kernel density estimates (KDEs) and all plots
xmax = 4000 #upper limit for PDPs and KDEs and all plots
xint = 1 # discretization interval for PDPs and KDEs only

x = np.arange(xmin,xmax+1,xint) # define discretized x axis

def ecdf(data): # function for making cumulative distribution functions (CDFs)
    cdfx = np.sort(data)
    cdfn = cdfx.size
    cdfy = np.arange(1,cdfn+1)/n
    return(cdfx,cdfy)

PDPs = np.zeros((len(x),N)) # preallocate PDP variable
CDFx = np.zeros((len(df),1)) # preallocate CDF x axis, this is only for plottting in the loop and is not needed after 
CDFy = np.zeros((len(df),1)) # preallocate CDF x axis, this is only for plottting in the loop and is not needed after 

fig, axs = plt.subplots(2,1,sharex=True) #set up figure, probably is a better/more efficient/prettier way to do this!

# loop thru the individual samples to make all PDPs and plot all CDFs
for j in range(N):
    data = df.iloc[:,j*2:j*2+2] #each sample is a set of 2 columns, so use this to loop thru and select pairs of columns
    data = data.apply(pd.to_numeric,errors='coerce')
    data = data.dropna() #remove N/A
    m = data.iloc[:,0] # individual sample ages
    s = data.iloc[:,1] # individual sample uncertainties
    n = len(m)
    f = np.zeros((len(x),len(m))) #preallocate matrix for individual Gaussian curves
    for i in range(n): # make PDPs, probably should def this as a separate function
        f[:,i] = (1/(s[i]*np.sqrt(2*np.pi))*np.exp((-((x-m[i])**2))/(2*((s[i])**2)))*xint) # make gaussians for each age
    PDPs[:,j] = f.sum(axis=1)/len(m) #sum Gaussians and normalize for each sample
    cdfx,cdfy = ecdf(m) #make CDF
    axs[0].plot(cdfx, cdfy) #plot CDFs on top of each other
axs[1].plot(x, PDPs)
plt.xlabel('Age (Ma)')

# KS Test is the max absolute difference btw 2 CDF curves
def KSTest(data1, data2):
    (data1, data2) = (np.ma.asarray(data1), np.ma.asarray(data2))
    (n1, n2) = (data1.count(), data2.count())
    mix = np.ma.concatenate((data1.compressed(), data2.compressed()))
    mixsort = mix.argsort(kind='mergesort')
    csum = np.where(mixsort<n1,1./n1,-1./n2).cumsum()
    KSTestD = max(np.abs(csum))
    return (KSTestD)

#Kuiper test is the sum of the max difference of CDF1 - CDF2 and CDF2 - CDF1
def KuiperTest(data1, data2):
    (data1, data2) = (np.ma.asarray(data1), np.ma.asarray(data2))
    (n1, n2) = (data1.count(), data2.count())
    mix = np.ma.concatenate((data1.compressed(), data2.compressed()))
    mixsort = mix.argsort(kind='mergesort')
    csum = np.where(mixsort<n1,1./n1,-1./n2).cumsum()
    KuiperTestV = max(csum) + max(csum*-1)
    return (KuiperTestV)

# Similarity (Gehrels, 2000) is the sum of the geometric mean of each point along x for two PDPs or KDEs
def SimilarityTest(data1,data2):
    Similarity = np.sum(np.sqrt(data1*data2))
    return(Similarity)

# Likeness (Satkoski et al., 2013) is the complement to Mismatch (Amidon et al., 2005) and is the sum of the 
# absolute difference divided by 2 for every pair of points along x for two PDPs or KDEs
def LikenessTest(data1,data2):
    Likeness = 1 - np.sum(abs(data1-data2))/2
    return(Likeness)

# Cross-correlation is the coefficient of determination (R squared), the simple linear regression between two PDPs or KDEs
def CrossCorrelationTest(data1,data2):
    correlation_matrix = np.corrcoef(data1, data2)
    correlation_xy = correlation_matrix[0,1]
    CrossCorrelation = correlation_xy**2
    return(CrossCorrelation)

KSTestD = np.zeros((N,N)) #Allocate pairwise comparison matrix for KS D statistic
KuiperTestV = np.zeros((N,N)) #Allocate pairwise comparison matrix for Kuipet Test V statistic
Similarity = np.zeros((N,N)) #Allocate pairwise comparison matrix for Similarity
Likeness = np.zeros((N,N)) #Allocate pairwise comparison matrix for Likeness
CrossCorrelation = np.zeros((N,N)) #Allocate pairwise comparison matrix for Cross-correlation

#loop thru comparison of every sample vs every sample, ijncluding vs themselves
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