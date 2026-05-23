import matplotlib.pyplot as plt
import numpy as np
x = [1,2,3,4,5]
y = [1,4,9,16,25]
plt.plot(x,y , color="red",linestyle="-",marker = "o",linewidth=2,markersize=10,markerfacecolor = "yellow")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Line")
plt.show()
