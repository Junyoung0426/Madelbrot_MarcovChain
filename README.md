# AMS_325-Python
* Python Code
  * mandelbrot.py
  * markov_chain.py

### madelbrot.py 

``` python
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# Make function that takes n, N_max, and threshold as input,

def mandelbrot(N_max, threshold, n):
    # Construct an n×n grid (2-dimension array) of points (x, y) in range [−2, 1]×[−1.5, 1.5]
    x = np.linspace(-2, 1, n)
    y = np.linspace(-1.5, 1.5, n)
    x, y = np.meshgrid(x, y)
    # Corresponding complex values c = x + yi (imaginary unit in Python is 1j)
    c = x+1j*y
    z = 0
    # Perform the iteration to compute z for each complex value in the grid
    
    for j in range(N_max):
        z = z**2 + c
    # Form a 2 dimension boolean array which name is mask indicating which points are in the set using, |z| < threshoold)
    mask = np.array((abs(z) < threshold))
    plt.imshow(mask, extent = np.array([-2, 1, -1.5, 1.5]))
    plt.gray()
    plt.savefig('mandelbrot.png')


# ---- Test the function with different n and see the image ----
#mandelbrot(N_max =50, threshold = 50., n =100)
#mandelbrot(N_max =50, threshold = 50., n =300)
mandelbrot(N_max=50, threshold=50., n=500)
#mandelbrot(N_max =50, threshold = 50., n =1000)
#mandelbrot(N_max =50, threshold = 50., n =3000)

```
To run this code, I will recommend you use Jupyter Notebook or Visual Studio Code or Pycharm.
This code shows the Mandelbrot fractal with the following Mandelbrot iteration on each point.
For me, I use Jupyter Notebook, make a new notebook and copy and paste the code.
When you run mandelbrot(N_max=50, threshold=50., n=500), you can draw the png photo which will be saved with the name,'mandelbrot.png'.
There will be differences when the input n was changed.


### markov_chain.py
``` python
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
    eig_val, eig_vec = linalg.eig(P.T)
    p_stationary = np.array(eig_vec[:,np.argmax(eig_val)])
    # Rescale the entries of the eigenvector so that its sum is equal to 1. Let the resulting vector be pstationary
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
        p_norm_array = np.append(p_norm_array,np.array(p_norm))
        count_array = np.append(count_array,np.array(j))
    print(p_norm_array)

    # make the plot the norms against i
    plt.plot(count_array, p_norm_array, 'ro-')
    # make x label, y label and title in plot
    plt.xlabel("i steps")
    plt.ylabel("norm of p-p_stationary")
    plt.title("steps vs norm of p-p_stationary")

    
# Test the function with some other values of n and N.
    
#markov_chain(n = 5, N =50)
#markov_chain(n = 10, N = 100)
markov_chain(n = 5, N = 60)
#markov_chain(n =5, N= 70)
#markov_chain(n = 5,N = 80)
``` 
To run this code, I will recommend you use Jupyter Notebook or Visual Studio Code or Pycharm.
For me, I use Jupyter Notebook, make a new notebook and copy and past the code.
This code shows the markov_chain. 
When you save this code and run markov_chain(n = 10, N = 100), it will print the plot which is a number of steps vs the norm of p-p_stationary.
You can change the inputs, n, and N, and will see a different plot.

