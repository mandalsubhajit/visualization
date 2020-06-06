import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap


def nested_circles(data, labels=None, c=None, ax=None, 
                   cmap=None, norm=None, textkw={}):
    ax = ax or plt.gca()
    data = np.array(data)
    R = np.sqrt(data/data.max())
    p = [plt.Circle((0,r), radius=r) for r in R[::-1]]
    arr = data[::-1] if c is None else np.array(c[::-1])
    col = PatchCollection(p, cmap=cmap, norm=norm, array=arr)
    
    ax.add_collection(col)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.autoscale()
    
    if labels is not None:
        kw = dict(color="white", va="center", ha="center")
        kw.update(textkw)
        ax.text(0, R[0], labels[0], **kw)
        for i in range(1, len(R)):
            ax.text(0, R[i]+R[i-1], labels[i], **kw)
    return col


# Data & Labels
data = [1,3,4,5,6]
labels = list("ABCDE")

# transforming the minimum of the color range from "white" to "light blue" so that the white text is visible for the innermost circle
blues = plt.cm.get_cmap('Blues', 256)
newcolors=blues(np.linspace(0, 1, 256))
newcmp = ListedColormap(newcolors[80:, :])

nested_circles(data, labels=labels, cmap=newcmp, textkw=dict(fontsize=14))

plt.savefig('nested_circles.png', dpi=300, transparent=True)
plt.close()

