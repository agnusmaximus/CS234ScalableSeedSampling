import sys
import numpy as np
import json
import matplotlib.pyplot as plt

def load_data(fname):
    print(fname)
    with open(fname, "r") as f:
        lines = f.readlines()
        return eval(lines[-1])

def load_data_avg(v):
    ds = []
    for fname in v:
        ds.append({x[0]:x[1] for x in load_data(fname)})

    ds = sorted(ds, key=lambda x:-len(x.keys()))
    d_final = {k:[] for k in ds[0].keys()}
    for d in ds:
        for k in d_final.keys():
            if k in d:
                d_final[k].append(d[k])
            else:
                d_final[k].append(200)

    d_final_tuple = ((k,np.mean(v)) for k,v in d_final.items())
    return sorted(d_final_tuple, key=lambda x:x[0])

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

files = {1:["1_out_%d" % i for i in range(1,11)],
         2:["2_out_%d" % i for i in range(1,11)],
         4:["4_out_%d" % i for i in range(1,11)],
         8:["8_out_%d" % i for i in range(1,11)]}
data = {k:load_data_avg(v) for k,v in files.items()}
plot_data(data)
