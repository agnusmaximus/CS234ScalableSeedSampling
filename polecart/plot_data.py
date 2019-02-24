import sys
import json
import matplotlib.pyplot as plt

def load_data(fname):
    print(fname)
    with open(fname, "r") as f:
        lines = f.readlines()
        return eval(lines[-1])

def plot_data(data):
    workers = sorted(data.keys())
    for worker in workers:
        legend = "par=%d" % worker
        plot_individual_datum(data[worker], legend)
    plt.title("Cart Pole with Simulated Parallelism")
    plt.xlabel("Number of iterations")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.savefig("plot.png")

def plot_individual_datum(data, name):
    x_values = [d[0] for d in data]
    y_values = [d[1] for d in data]
    plt.plot(x_values, y_values, label=name)

files = {1:"1_out", 2:"2_out", 4:"4_out", 8:"8_out"}
data = {k:load_data(v) for k,v in files.items()}
plot_data(data)
