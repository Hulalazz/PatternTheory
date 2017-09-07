# generation, storage and use of of random variables
# so that you can use the same RNG sample for various plotting

import numpy as np


def randn(size):
    if size == 100:
        return np.loadtxt("randvect/randn100", delimiter='\n')
    elif size == 1000:
        return np.loadtxt("randvect/randn1000", delimiter='\n')
    elif size == 2000:
        return np.loadtxt("randvect/randn2000", delimiter='\n')
    elif size == 4000:
        return np.loadtxt("randvect/randn4000", delimiter='\n')
    elif size == 6000:
        return np.loadtxt("randvect/randn6000", delimiter='\n')
    elif size == 10000:
        return np.loadtxt("randvect/randn10000", delimiter='\n')
    elif size == 22049:
        return np.loadtxt("randvect/randn22049", delimiter='\n')
    elif size == 44100:
        return np.loadtxt("randvect/randn44100", delimiter='\n')
    else:
        print("error")
        return 0


def randtime(size):
    if size == 1000:
        return np.loadtxt("randvect/randtime1000", delimiter='\n')
    elif size == 4000:
        return np.loadtxt("randvect/randtime4000", delimiter='\n')
    else:
        print("error")
        return 0


def random(size):
    if size == 100:
        return np.loadtxt("randvect/random100", delimiter='\n')
    elif size == 1000:
        return np.loadtxt("randvect/random1000", delimiter='\n')
    elif size == 2000:
        return np.loadtxt("randvect/random2000", delimiter='\n')
    elif size == 4000:
        return np.loadtxt("randvect/random4000", delimiter='\n')
    elif size == 6000:
        return np.loadtxt("randvect/random6000", delimiter='\n')
    elif size == 10000:
        return np.loadtxt("randvect/random10000", delimiter='\n')
    elif size == 22049:
        return np.loadtxt("randvect/random22049", delimiter='\n')
    elif size == 44100:
        return np.loadtxt("randvect/random44100", delimiter='\n')
    else:
        print("error")
        return 0



# do not uncomment all of them, will change values

'''
#np.savetxt("randvect/randn100", np.random.randn(100), delimiter='\n')
#np.savetxt("randvect/randn1000", np.random.randn(1000), delimiter='\n')
#np.savetxt("randvect/randn2000", np.random.randn(2000), delimiter='\n')
#np.savetxt("randvect/randn4000", np.random.randn(4000), delimiter='\n')
#np.savetxt("randvect/randn6000", np.random.randn(6000), delimiter='\n')
#np.savetxt("randvect/randn10000", np.random.randn(10000), delimiter='\n')
#np.savetxt("randvect/randn22049", np.random.randn(22049), delimiter='\n')
#np.savetxt("randvect/randn44100", np.random.randn(44100), delimiter='\n')


np.savetxt("randvect/random100", np.random.random(100), delimiter='\n')
np.savetxt("randvect/random1000", np.random.random(1000), delimiter='\n')
np.savetxt("randvect/random2000", np.random.random(2000), delimiter='\n')
np.savetxt("randvect/random4000", np.random.random(4000), delimiter='\n')
np.savetxt("randvect/random6000", np.random.random(6000), delimiter='\n')
np.savetxt("randvect/random10000", np.random.random(10000), delimiter='\n')
np.savetxt("randvect/random22049", np.random.random(22049), delimiter='\n')
np.savetxt("randvect/random44100", np.random.random(44100), delimiter='\n')
'''
