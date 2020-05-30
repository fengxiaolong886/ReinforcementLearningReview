import matplotlib
from matplotlib import pyplot as plt

def plot_3D(X, Y, Z, xlabel, ylabel, zlabel, title):
    fig = plt.figure(figsize=(20, 10), facecolor = "white")
    ax = fig.add_subplot(111, projection = "3d")
    surf = ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, 
                           cmap=matplotlib.cm.rainbow, vmin=-1.0, vmax=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    ax.set_facecolor("white")
    fig.colorbar(surf)
    return fig

