import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg


def markov_chain(n,N):
    # I use np.array instead of List to utilize the numpy
    # Construct a random n-vector with non-negative entries and scale its entries so that the sum is 1. 
    # This computation gives us a probability distribution p.
    p = np.array(np.random.rand(n))
    p /= p.sum()  
    print(p)
    
    # Construct a random n × n (say n = 5) matrix with non-negative entries, and scale the entries so that
    # the sum for each row is 1. This computation gives us a transition matrix P.
    P = np.array(np.random.rand(n,n))     
    P /= P.sum(axis=1)[:,np.newaxis]
    print(P)

    # Use the function np.linalg.eig to compute the eigenvector of P.T corresponding to the largest eigenvalue.
    eig_val, eig_vec = np.linalg.eig(P.T)
    p_stationary = np.array(eig_vec[:,np.argmax(eig_val)])
    # Rescale the entries of the eigenvector so that its sum is equal to 1.  Let the resulting vector be pstationary
    p_stationary /= p_stationary.sum()
    print(p_stationary)

    #make arrays for the p_norm and number of steps
    p_norm_array = np.array([])
    count_array = np.array([])
    
    for j in range(N):
        # Starting from p as the initial state, compute the transition for N (say N = 50) steps 
        # compute p ← P.T.dot(p) for N times
        p = P.T.dot(p)
        # compute the norm of p − pstationary
        p_norm = np.array(linalg.norm(np.subtract(p,p_stationary)))
        p_norm_array = np.append(p_norm_array,p_norm)
        count_array = np.append(count_array,j)
    print(p_norm_array)
    print(count_array)

    # make the plot the norms against i
    plt.plot(count_array, p_norm_array, 'ro-')
    # make x label, y label and title in plot
    plt.xlabel("i steps")
    plt.ylabel("norm of p-p_stationary")
    plt.title("steps vs norm of p-p_stationary")

    
# Test the function with some other values of n and N.
#markov_chain(n = 5, N = 10)
markov_chain(n = 5, N =50)
#markov_chain(n = 5, N = 100)
#markov_chain(n = 10, N = 50)
#markov_chain(n = 10, N = 100)