import numpy as np
    
def pdp(data,xmin,xmax,xint):
    x = np.arange(xmin,xmax+1,xint) # define discretized x axis
    m = data.iloc[:,0] # individual sample ages
    s = data.iloc[:,1] # individual sample uncertainties
    n = len(m)
    f = np.zeros((len(x),len(m))) #preallocate matrix for individual Gaussian curves
    for i in range(n): # make PDPs, probably should def this as a separate function
        f[:,i] = (1/(s[i]*np.sqrt(2*np.pi))*np.exp((-((x-m[i])**2))/(2*((s[i])**2)))*xint) # make gaussians for each age
    PDP = f.sum(axis=1)/len(m) #sum Gaussians and normalize for each samples
    return(PDP)

def ecdf(data,xmin,xmax,xint): # function for making cumulative distribution functions (CDFs)
    cdfx = np.sort(data.iloc[:,0])
    cdfn = cdfx.size
    cdfy = np.arange(1,cdfn+1)/len(cdfx)
    return(cdfx,cdfy)