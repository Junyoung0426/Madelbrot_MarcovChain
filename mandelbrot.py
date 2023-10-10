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
mandelbrot(N_max =50, threshold = 50., n =200)

#mandelbrot(N_max =50, threshold = 50., n =100)
#mandelbrot(N_max =50, threshold = 50., n =300)
#mandelbrot(N_max=50, threshold=50., n=500)
#mandelbrot(N_max =50, threshold = 50., n =1000)
#mandelbrot(N_max =50, threshold = 50., n =3000)
