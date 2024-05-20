import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
particles = [pd.DataFrame({"P.X": [], "P.Y": [], "V.X": [], "V.Y": []}) for i in range(100)]
data = "../results.csv"
with open(data, newline = "\n") as f:
    reader = csv.reader(f, delimiter = ",")
    for row in reader:
        if row != []:
            particles[int(row[0])].loc[len(particles[int(row[0])])] = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)

def animate(i):
    ax.clear()
    for j in range(len(particles)):
        point = (particles[j].loc[i].iloc[0], particles[j].loc[i].iloc[1])
        ax.plot(point[0], point[1], color = "green", label = "original", marker = "o")
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
ani = animation.FuncAnimation(fig, animate, frames = len(particles[0]), interval = 16.7, repeat = True)
ani.save("ani.gif", dpi = 200, writer = "pillow", fps = 60)
plt.show()