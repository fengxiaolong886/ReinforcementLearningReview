import matplotlib.pyplot as plt
import numpy as np
from constants import *
import os

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_and_save(content,path,name):
    check_path(path)    
    savename=os.path.join(path,name)
    plt.plot(np.arange(len(content)),content)
    plt.ylabel("Cost")
    plt.xlabel("steps")
    plt.savefig(savename)
    plt.show()
