import numpy as np

# KS Test (Massey, 1951) is the max absolute difference btw 2 CDF curves 
def KSTest(data1, data2):
    (data1, data2) = (np.ma.asarray(data1), np.ma.asarray(data2))
    (n1, n2) = (data1.count(), data2.count())
    mix = np.ma.concatenate((data1.compressed(), data2.compressed()))
    mixsort = mix.argsort(kind='mergesort')
    csum = np.where(mixsort<n1,1./n1,-1./n2).cumsum()
    KSTestD = max(np.abs(csum))
    return (KSTestD)

#Kuiper test (Kuiper, 1960) is the sum of the max difference of CDF1 - CDF2 and CDF2 - CDF1 
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